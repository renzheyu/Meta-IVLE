import os
import numpy as np
import pandas as pd
import pickle
from functools import reduce
from itertools import product
from src.utils import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import robust_scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.proportion import proportions_ztest


def create_master_table(hdf_in, hdf_out, out_dir, max_var_miss, label_table_names,
                        protected_table_name='protected_attributes', table_names=None, merge_how='outer',
                        standardize=False, by=None, to_csv=True):
    """
    Merge and clean separate feature tables into a master table

    Parameters
    ----------
    hdf_in : HDFStore object
        Where the feature tables are stored (comparable to a schema in databases)

    hdf_out : HDFStore object
        Where the resulting master table is stored (comparable to a schema in databases)

    out_dir : str
        Directory to save the resulting table

    max_var_miss : float
        If a record is missing on more than this proportion of features, drop it

    label_table_names : list
        List of tables that contain labels/outcome variables

    protected_table_name : string
        Table that contains protected attributes

    table_names : list
        List of tables (stored in hdf) to be merged; if None, all tables in hdf will be merged
        Assuming that all tables in the list have same indices to merge on

    merge_how : {'left', 'right', 'outer', 'inner'}, default 'outer'
        Type of merge to be performed, passed to 'how' parameter in pd.merge function

    standardize : boolean
        Whether to normalize (continuous) features (within course)

    by : list
        List of variables that define the group within which features are standardized, can be

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    master_table : Pandas DataFrame
        Combined feature/label table with two levels of columns, where the first level identifies the original table
        names (e.g., 'demographics') and the second level includes the columns from the original tables
    """
    print('Reading feature tables and creating master table')
    if table_names is None:
        table_names = [key.strip('/') for key in hdf_in.keys()]
    table_list = []
    for tname in table_names:
        table = hdf_in[tname]
        table.columns = pd.MultiIndex.from_product([[tname], table.columns])
        table_list.append(table)
    master_table = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how=merge_how), table_list)
    # Remove rows that are missing in any labels
    label_miss = master_table[label_table_names].isnull().any(axis=1)
    # Remove rows that have too many missing features (not including protected attributes)
    over_var_miss = (master_table.drop(protected_table_name, axis=1).isnull().sum(axis=1) >=
                     max_var_miss * len(master_table.drop(protected_table_name, axis=1).columns))
    master_table = master_table[~(label_miss | over_var_miss)]
    if standardize:
        bool_cols, non_bool_cols = split_bool_cols(master_table)
        non_bool_unprotected_cols = [col for col in non_bool_cols if col[0] != protected_table_name]
        master_table[non_bool_unprotected_cols] = master_table.groupby(by)[non_bool_unprotected_cols].transform(
            robust_scale)

    hdf_out.put('master_table', master_table)
    print('Master table info saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'master_table.csv')
        master_table.to_csv(csv_path)
        print(f'Master table saved to {csv_path}')

    return master_table


def get_features(master_table, feature_names, to_numpy=False):
    """
    Get feature matrix from master_table according to the given feature names

    Parameters
    ----------
    master_table : Pandas DataFrame
        Combined feature/label table with two levels of columns, where the first level identifies the feature/label
        groups and the second level includes the specific features/labels

    feature_names : str
        Textual identification of one or more feature groups in the form of 'grpA+grpB'

    to_numpy : boolean
        Whether to transform resulting Pandas DataFrame to a plain numpy array

    Returns
    -------
    feature_table : Pandas DataFrame
    """
    feature_groups = feature_names.split('+')
    feature_table = master_table[[f'{fgroup}_features' for fgroup in feature_groups]]
    if to_numpy:
        feature_table = feature_table.to_numpy()
    return feature_table


def get_labels(master_table, label_name, label_group='labels', to_numpy=False):
    """
    Get label array from master_table according to the given label name, only one at a time

    Parameters
    ----------
    master_table : Pandas DataFrame
        Combined feature/label table with two levels of columns, where the first level identifies the feature/label
        groups and the second level includes the specific features/labels

    label_name : str
        Label name as a column name in master_table

    label_group : str
        The label group that label_name is under

    to_numpy : boolean
        Whether to transform resulting Pandas DataFrame to a plain numpy array

    Returns
    -------
    label_table : Pandas DataFrame
    """
    label_table = master_table[label_group][label_name]
    if to_numpy:
        label_table = label_table.to_numpy()
    return label_table


def run_pred_models(master_table, features, labels, models, group_var, rseed, model_dir, result_dir, result_hdf,
                    imputer='iterative', tune_models=False, to_csv=True):
    """
    Configure all requested prediction models (as combinations of different features, labels and models),
    run the models using group(course)-level cross validation and save raw predictions

    Parameters
    ----------
    master_table : Pandas DataFrame
        Combined feature/label table with two levels of columns, where the first level identifies the feature/label
        groups and the second level includes the specific features/labels

    features : list
        List of feature configurations, each being textual identification of one or more feature groups in the form
        of 'grpA+grpB'

    labels : list
        List of label names, as columns in master_table

    models : dict
        Dictionary of models along with their associated hyperparameters to do grid search on

    group_var : str
        Variable name to group samples for cross validation. Currently must be in the index of master_table

    rseed : int
            Pseudo-random number

    imputer : str
        Name of imputer to use; either "simple", "KNN" or "iterative"

    model_dir : str
        Directory to save model hyperparameters

    result_dir : str
        Directory to save the resulting table

    result_hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    tune_models : boolean
        Whether to perform grid-search to find (and save) best hyperparameters, or to use saved hyperparameters

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    model_info : Pandas DataFrame
        model_id | feature | label | model

    pred_res : Pandas DataFrame
        student/course_id | model_id | y_true | y_pred
    """
    # Construct empty model info table and add models row by row
    model_info = pd.DataFrame([], columns=['model_id', 'feature', 'label', 'model'])
    unique_id = master_table.index.to_frame().reset_index(drop=True)
    pred_res = pd.DataFrame([], columns=unique_id.columns.tolist()+['model_id', 'y_true', 'y_pred'])
    best_params_file_path = os.path.join(model_dir, 'best_hyperparams.pickle')
    clf_dict = {
        'logistic_regression': LogisticRegression(random_state=rseed, solver='liblinear'),
        'svm': SVC(probability=True, random_state=rseed),
        'random_forest': RandomForestClassifier(random_state=rseed)
    }
    imputer_dict = {
        'simple': SimpleImputer(strategy='constant', fill_value=0),
        'KNN' : KNNImputer(),
        'iterative': IterativeImputer()
    }

    if not tune_models:
        with open(best_params_file_path, 'rb') as f:
            hyperparams = pickle.load(f)
    else:
        hyperparams = dict()

    for (feature, label) in product(features, labels):
        X = get_features(master_table, feature).to_numpy()
        y = get_labels(master_table, label).to_numpy()
        groups = master_table.index.get_level_values(group_var)
        for model in models:
            model_id = len(model_info) + 1
            print(f'Predicting {label} using {feature} via {model}...')
            clf = clf_dict[model]
            logo = LeaveOneGroupOut()
            estimator = Pipeline(
                [('impute', imputer_dict[imputer]),
                 ('clf', clf)]
            )
            if not tune_models:
                best_params = hyperparams.get((feature, label, model))
            else:
                params_grid = models.get(model)
                params_grid = {'clf__'+param: params_grid[param] for param in params_grid}
                clf_tuned = GridSearchCV(estimator, params_grid, cv=logo, n_jobs=3)
                clf_tuned.fit(X, y, groups=groups)
                best_params = clf_tuned.best_params_
                hyperparams[(feature, label, model)] = best_params
            print(best_params)
            estimator.set_params(**best_params)
            predicted = cross_val_predict(estimator, X, y, groups=groups, cv=logo)
            predicted_proba = cross_val_predict(estimator, X, y, groups=groups, cv=logo, method='predict_proba')[
                              :, 1]
            model_info = model_info.append({'model_id': model_id, 'feature': feature, 'label': label, 'model': model},
                              ignore_index=True)
            res = pd.DataFrame({'model_id': model_id, 'y_true': y, 'y_pred': predicted, 'y_proba': predicted_proba})
            pred_res = pred_res.append(pd.concat([unique_id, res], axis=1))

    if tune_models:
        with open(best_params_file_path, 'wb') as f:
            pickle.dump(hyperparams, f)
        print("Tuned hyperparameters saved to pickle file")

    result_hdf.put('model_info', model_info)
    print('Model info saved to HDFStore')
    result_hdf.put('pred_res', pred_res)
    print('Raw prediction results saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(result_dir, 'model_info.csv')
        model_info.to_csv(csv_path, index=False)
        print(f'Model info saved to {csv_path}')
        csv_path = os.path.join(result_dir, 'pred_res.csv')
        pred_res.to_csv(csv_path, index=False)
        print(f'Raw prediction results saved to {csv_path}')

    return model_info, pred_res


def eval_pred_res(pred_res, metrics, out_dir, hdf, to_csv=True):
    """
    Evaluate raw prediction results based on given metrics

    Parameters
    ----------
    pred_res : Pandas DataFrame
        Raw prediction results, one row per model per entity
        Format:
            entity_id | model_id | y_true | y_pred

    metrics : list
        List of metric names

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Results
    -------
    pred_eval_score : Pandas DataFrame
        Evaluation score of prediction results, one row per model
        Format:
            model_id | tp | tn | fp | fn | f1 | metric1 | metric2 | ...
    """
    pred_eval_score = pred_res.groupby('model_id').apply(lambda x: pd.DataFrame(
        {
            'tp': [len(x.query('y_true == 1 and y_pred == 1'))],
            'tn': [len(x.query('y_true == 0 and y_pred == 0'))],
            'fp': [len(x.query('y_true == 0 and y_pred == 1'))],
            'fn': [len(x.query('y_true == 1 and y_pred == 0'))]
        }
    )).droplevel(1).apply(pd.to_numeric)

    pred_eval_score = pred_eval_score.assign(f1=lambda x: 2*x['tp']/(2*x['tp']+x['fp']+x['fn']))
    for metric in metrics:
        print(f'Calculating {metric}...')
        if metric == 'acc':
            pred_eval_score = pred_eval_score.assign(acc=lambda x: (x['tp']+x['tn'])/(x['tp']+x['tn']+x['fp']+x['fn']))
        elif metric == 'fnr':
            pred_eval_score = pred_eval_score.assign(fnr=lambda x: x['fn']/(x['fn']+x['tp']))
        elif metric == 'fpr':
            pred_eval_score = pred_eval_score.assign(fpr=lambda x: x['fp']/(x['fp']+x['tn']))
    pred_eval_score = pred_eval_score.reset_index()

    hdf.put('pred_score', pred_eval_score)
    print('Prediction scores saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'pred_score.csv')
        pred_eval_score.to_csv(csv_path, index=False)
        print(f'Prediction scores saved to {csv_path}')
    return pred_eval_score


def compare_pred_score(pred_eval_score, out_dir, hdf, to_csv=True):
    """
    Parameters
    ----------
    pred_eval_score : Pandas DataFrame
        Evaluation score of prediction results, one row per model
        Format:
            model_id | tp | tn | fp | fn | f1 | metric1 | metric2 | ...

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Results
    -------
    model_comp : Pandas DataFrame
        Results of statistical tests of pairwise model result comparisons
        Format:
            model_id_1 | model_id_2 | diff_metric1 | p_val_metrics1 | ...
    """
    # TODO: write this function when necessary


def compute_bias(df, ref_groups, metrics, one_sided=False):

    """
    Compute the bias of a model

    Parameters
    ----------
    df : Pandas DataFrame
        Raw prediction results concatenated with protected attribute information
        Ex:
            entity_id | model_id | y_true | y_pred | ethnicity | hs_GPA

    metrics : list
        List of metrics to be calculated

    ref_groups : dict
        Identifies the reference group for each protected attribute

    one_sided : boolean
        Whether to perform one-sided (or two-sided) tests when calculating biases between groups

    Results
    -------
    bias : Pandas DataFrame
        Bias measures against different groups for a single model
        Format:
            attribute_name | attribute_value | bias1_disparity | bias1_significance | ...
     """

    columns = []
    for metric in metrics:
        columns.append(metric)
        columns.append(metric + "_disparity")
        columns.append(metric + "_significance")

    total_n = len(df)
    bdf = pd.DataFrame([], columns=['attribute_name', 'attribute_value', 'total_n', 'group_n', 'label_pos', 'label_neg', 'pp', 'pn', 'fp', 'fn', 'tn', 'tp', 'ref_group_value'] + columns)
    for att in ref_groups:
        attribute_name = att
        df_att = df.groupby(att)
        groups = list(df[att].sort_values().unique())
        ref_group = ref_groups[att]
        groups = [ i for i in groups if str(i) != 'nan']
        df_ref = df_att.get_group(ref_group)

        fp_ref = len(df_ref.query('y_true == 0 and y_pred == 1'))
        fn_ref = len(df_ref.query('y_true == 1 and y_pred == 0'))
        tn_ref = len(df_ref.query('y_true == 0 and y_pred == 0'))
        tp_ref = len(df_ref.query('y_true == 1 and y_pred == 1'))

        for metric in metrics:
            if metric == 'acc':
                ref_acc = (tp_ref + tn_ref)/(tp_ref + tn_ref + fp_ref + fn_ref)
            elif metric == 'fnr':
                ref_fnr = fn_ref/(fn_ref + tp_ref)
            elif metric == 'fpr':
                ref_fpr = fp_ref/(fp_ref + tn_ref)

        for group in groups:
            attribute_value = group
            df_group = df_att.get_group(group)
            group_n = len(df_group)
            label_pos = len(df_group[df_group['y_true'] == 1])
            label_neg = group_n - label_pos
            pp = len(df_group[df_group['y_pred'] == 1])
            pn = group_n - pp
            fp = len(df_group.query('y_true == 0 and y_pred == 1'))
            fn = len(df_group.query('y_true == 1 and y_pred == 0'))
            tn = len(df_group.query('y_true == 0 and y_pred == 0'))
            tp = len(df_group.query('y_true == 1 and y_pred == 1'))

            metric_dict = dict()
            for metric in metrics:
                if metric == 'acc':
                    value = (tp + tn)/(fp + fn + tn + tp)
                    disparity = min(ref_acc/(value + 0.00001), 10)
                    count = [tp + tn, tp_ref + tn_ref]
                    nobs = [fp + fn + tn + tp, tp_ref + tn_ref + fp_ref + fn_ref]
                    alternative = 'smaller' if one_sided else 'two-sided'
                    stat, pval = proportions_ztest(count, nobs, alternative=alternative)
                elif metric == 'fnr':
                    value = fn/(fn + tp) if (fn + tp) != 0 else np.nan
                    disparity = min(value/(ref_fnr + 0.00001), 10)
                    count = [fn, fn_ref]
                    nobs = [fn + tp, fn_ref + tp_ref]
                    alternative = 'larger' if one_sided else 'two-sided'
                    stat, pval = proportions_ztest(count, nobs, alternative=alternative)
                elif metric == 'fpr':
                    value = fp/(fp + tn) if (fp + tn) != 0 else np.nan
                    disparity = min(value/(ref_fpr + 0.00001), 10)
                    count = [fp, fp_ref]
                    nobs = [fp + tn, fp_ref + tn_ref]
                    alternative = 'larger' if one_sided else 'two-sided'
                    stat, pval = proportions_ztest(count, nobs, alternative=alternative)

                metric_dict[metric] = value
                metric_dict[metric + "_disparity"] = disparity
                metric_dict[metric + "_significance"] = pval

            row = {
                'attribute_name': attribute_name,
                'attribute_value': attribute_value,
                'total_n': total_n,
                'group_n': group_n,
                'label_pos': label_pos,
                'label_neg': label_neg,
                'pp': pp, 'pn': pn,
                'fp': fp, 'fn': fn, 'tn': tn, 'tp': tp,
                'ref_group_value': ref_group
            }

            row.update(metric_dict)
            row = pd.DataFrame([row])

            bdf = bdf.append(row, ignore_index=True, sort=False)

    non_numeric_cols = ['attribute_name', 'attribute_value', 'ref_group_value']
    numeric_cols = bdf.columns.difference(non_numeric_cols)
    bdf[numeric_cols] = bdf[numeric_cols].apply(pd.to_numeric)

    return bdf


def audit_fairness(pred_res, protected_attrs, ref_groups, metrics, one_sided, out_dir, hdf, to_csv=True):
    """
    Evaluate the fairness of raw prediction results

    Parameters
    ----------
    pred_res : Pandas DataFrame
        Raw prediction results, one row per model per entity
        Ex:
            entity_id | model_id | y_true | y_pred

    protected_attrs : Pandas DataFrame
        Raw protected attributes
        Ex:
            entity_id | attr1 | attr2 | ...

    ref_groups : dict
        Identifies the reference group for each protected attribute

    one_sided : boolean
        Whether to perform one-sided (or two-sided) tests when calculating biases between groups

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Results
    -------
    bias : Pandas DataFrame
        Bias measures against different groups
        Format:
            model_id | attribute_name | attribute_value | bias1_disparity | bias1_significance | ...
    """
    id_cols = protected_attrs.index.to_frame().columns
    df = pred_res.merge(protected_attrs.reset_index()).drop(id_cols, axis=1)

    bias = df.groupby('model_id').apply(compute_bias, ref_groups=ref_groups, metrics=metrics,
                                        one_sided=one_sided).reset_index().drop('level_1', axis=1)

    hdf.put('pred_bias', bias)
    print('Prediction bias analysis saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'pred_bias.csv')
        bias.to_csv(csv_path, index=False)
        print(f'Prediction bias analysis saved to {csv_path}')


def run(feature_dir, model_dir, result_dir, model_config):
    """
    Run predictive models configured by the user and save results to the disk

    Parameters
    ----------
    feature_dir : str
        Directory where extracted features are stored

    model_dir : str
        Directory where best model hyperparameters are stored

    result_dir : str
        Directory where prediction results are stored

    model_config : str
        Name of model configuration file, with full path

    Returns
    -------
    None
    """

    model_configs = load_yaml(model_config)

    features = model_configs.get('features')
    labels = model_configs.get('labels')
    models = model_configs.get('models')
    metrics = model_configs.get('metrics')

    with pd.HDFStore(os.path.join(feature_dir, 'feature.h5')) as hdf_feature:
        with pd.HDFStore(os.path.join(result_dir, 'result.h5')) as hdf_result:
            # master_table = create_master_table(hdf_in=hdf_feature, hdf_out=hdf_result, out_dir=result_dir,
            #                                    max_var_miss=model_configs.get('max_var_miss'),
            #                                    label_table_names=['labels'], standardize=True, by=['course_id'])
            master_table = hdf_result['master_table']
            # model_info, pred_res = run_pred_models(master_table, features, labels, models, group_var='course_id',
            #                                        rseed=model_configs.get('random_seed'), model_dir=model_dir,
            #                                        result_dir=result_dir, result_hdf=hdf_result,
            #                                        tune_models=model_configs.get('tune_models'))
            pred_res = hdf_result['pred_res']
            # eval_pred_res(pred_res, metrics, out_dir=result_dir, hdf=hdf_result)
            audit_fairness(pred_res, protected_attrs=master_table['protected_attributes'],
                           ref_groups=model_configs.get('ref_groups'), metrics=metrics,
                           one_sided=model_configs.get('bias_test_one_sided'), out_dir=result_dir, hdf=hdf_result)