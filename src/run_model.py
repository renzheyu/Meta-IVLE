import os
import numpy as np
import pandas as pd
import pickle
from functools import reduce
from itertools import product
from src.utils import *

from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import robust_scale, Imputer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.proportion import proportions_ztest

from aequitas.group import Group
from aequitas.bias import Bias

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

def get_tuned_model(model_name, X, y, groups, params=None, param_grid=None, tune=False):
    """
    Configure a classifier instance given the input model name
    Currently no hyperparameter tuning but can accommodate in future versions

    Parameters
    ----------
    model_name : str

    X: Pandas DataFrame
        The feature matrix obtained from master table

    y: Pandas DataFrame
        The label array

    params : dict or None
        If not None, a dictionary of parameters for the specified classifier, in the form of {param_name: param_value}

    param_grid: dict or None
        If not None, a dictionary of parameters for the specified classifier that will be used for GridSearch

    Returns
    -------
    clf : Scikit-learn estimator
        Configured classifier
    """
    imputer = Imputer()
    x_imputed = imputer.fit_transform(X)

    indicator = MissingIndicator(features='all')
    x = indicator.fit_transform(x_imputed)

    clf_dict = {
        'logistic_regression': LogisticRegression(),
        'svm': SVC(probability=True),
        'random_forest': RandomForestClassifier()
    }
    clf = clf_dict[model_name]

    if tune and param_grid is not None:
       logo = LeaveOneGroupOut()
       clf_tuned = GridSearchCV(clf, param_grid, cv=logo)
       clf_tuned.fit(x, y, groups=groups)
       best_params = clf_tuned.best_params_
       print(best_params)
       predicted = clf_tuned.predict(x)
       predicted_proba = clf_tuned.predict_proba(x)[:,1]
       return predicted, predicted_proba, best_params

    if params is not None:
        clf.set_params(**params)

    return clf

def get_pred_res(master_table, features, labels, models, model_configs, group_var, out_dir, hdf, tune_models=False, to_csv=True):
    """
    Configure all requested prediction models (as combinations of different features, labels and models),
    run the models using group(course)-level ross validation and save raw predictions

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

    models : list
        List of classifier names

    model_configs: yaml
        Config file for models

    group_var : str
        Variable name to group samples for cross validation. Currently must be in the index of master_table

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

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

    if not tune_models:
        with open('./hyperparameters.pickle', 'rb') as f:
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

            if tune_models:
                param_grid = model_configs.get(model)
                predicted, predicted_proba, best_params = get_tuned_model(model, X, y, groups, param_grid=param_grid, tune=True)
                hyperparams[model_id] = best_params
            else:
                parameters = hyperparams[model_id]
                clf = get_tuned_model(model, X, y, groups, params=parameters)
                logo = LeaveOneGroupOut()
                estimator = make_pipeline(make_union(SimpleImputer(strategy='constant', fill_value=0), MissingIndicator(
                    features='all')), clf)
                predicted = cross_val_predict(estimator, X, y, groups=groups, cv=logo)
                predicted_proba = cross_val_predict(estimator, X, y, groups=groups, cv=logo, method="predict_proba")[:,1]

            model_info = model_info.append({'model_id': model_id, 'feature': feature, 'label': label, 'model': model},
                              ignore_index=True)
            res = pd.DataFrame({'model_id': model_id, 'y_true': y, 'y_pred': predicted, 'upper_level_score': predicted_proba})
            pred_res = pred_res.append(pd.concat([unique_id, res], axis=1))

    if tune_models:
        with open('hyperparameters.pickle', 'wb') as handle:
            pickle.dump(hyperparams, handle)
        print("Tuned hyperparameters saved to pickle file")

    hdf.put('model_info', model_info)
    print('Model info saved to HDFStore')
    hdf.put('pred_res', pred_res)
    print('Raw prediction results saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'model_info.csv')
        model_info.to_csv(csv_path, index=False)
        print(f'Model info saved to {csv_path}')
        csv_path = os.path.join(out_dir, 'pred_res.csv')
        pred_res.to_csv(csv_path, index=False)
        print(f'Raw prediction results saved to {csv_path}')

    return model_info, pred_res

def eval_pred_res(pred_res, metrics, out_dir, hdf, comp_model=False, to_csv=True):
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

    comp_model : boolean
        If true, compare each pair of models and generate and return a comparison table

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Results
    -------
    pred_eval_hits : Pandas DataFrame (to be deprecated)
        Raw prediction results concatenated with binary evaluation of these results given each metric
        For each metric, 1 assigned to predictions contributing to the numerator of its calculation, 0 to those
        contributing to the denominator, and NaN to those not included in the calculation (Ex. false positive rate =
        false positives / true negatives)
        Format:
            entity_id | model_id | metric1_ind | metric2_ind | ...

    pred_eval_score : Pandas DataFrame
        Evaluation score of prediction results, one rwo per model
        Format:
            model_id | metric1 | metric2 | ...

    model_comp : Pandas DataFrame (currently not used)
        Results of statistical tests of pairwise model result comparisons
        Format:
            model_id_1 | model_id_2 | diff_metric1 | p_val_metrics1 | ...
    """
    pred_eval_hits = pred_res.drop(['y_true', 'y_pred'], axis=1)
    for metric in metrics:
        print(f'Calculating {metric}...')
        pred_eval_hits[metric] = get_pred_eval_hits(metric, pred_res['y_true'], pred_res['y_pred'])
    pred_eval_score = pred_eval_hits.groupby('model_id')[metrics].mean().reset_index()
    pred_eval_hits[metrics] = pred_eval_hits[metrics].add_suffix('_ind')
    if comp_model:
        # TODO
        pass
    else:
        model_comp_res = None

    hdf.put('pred_score', pred_eval_score)
    print('Prediction scores saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'pred_score.csv')
        pred_eval_score.to_csv(csv_path, index=False)
        print(f'Prediction scores saved to {csv_path}')


def compute_bias(df, ref_groups, metrics):
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
        groups = list(df[att].unique())
        ref_group = ref_groups[att]
        groups = [ i for i in groups if str(i) != 'nan']
        df_ref = df_att.get_group(ref_group)

        fp_ref = len(df_ref.query('label_value == 0 and score == 1'))
        fn_ref = len(df_ref.query('label_value == 1 and score == 0'))
        tn_ref = len(df_ref.query('label_value == 0 and score == 0'))
        tp_ref = len(df_ref.query('label_value == 1 and score == 1'))

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
            label_pos = len(df_group[df_group['label_value'] == 1])
            label_neg = group_n - label_pos
            pp = len(df_group[df_group['score'] == 1])
            pn = group_n - pp
            fp = len(df_group.query('label_value == 0 and score == 1'))
            fn = len(df_group.query('label_value == 1 and score == 0'))
            tn = len(df_group.query('label_value == 0 and score == 0'))
            tp = len(df_group.query('label_value == 1 and score == 1'))

            metric_dict = dict()
            for metric in metrics:
                if metric == 'acc':
                    value = (tp + tn)/(fp + fn + tn + tp)
                    disparity = value/ref_acc if ref_acc != 0 else value/10
                    count = [tp + tn, tp_ref + tn_ref]
                    nobs = [fp + fn + tn + tp, tp_ref + tn_ref + fp_ref + fn_ref]
                    stat, pval = proportions_ztest(count, nobs, alternative='smaller')
                elif metric == 'fnr':
                    value = fn/(fn + tp) if fn != 0 else 0
                    disparity = value/ref_fnr if ref_fnr != 0 else value/10
                    count = [fn, fn_ref]
                    nobs = [fn + tp, fn_ref + tp_ref]
                    stat, pval = proportions_ztest(count, nobs, alternative='larger')
                elif metric == 'fpr':
                    value = fp/(fp + tn) if fp != 0 else 0
                    disparity = value/ref_fpr if ref_fpr != 0 else value/10
                    count = [fp, fp_ref]
                    nobs = [fp + tn, fp_ref + tn_ref]
                    stat, pval = proportions_ztest(count, nobs, alternative='larger')

                metric_dict[metric] = value
                metric_dict[metric + "_disparity"] = disparity
                metric_dict[metric + "_significance"] = pval

            row = {'attribute_name': attribute_name, 'attribute_value': attribute_value, 'total_n': total_n, 'group_n': group_n,
            'label_pos': label_pos, 'label_neg': label_neg, 'pp': pp, 'pn': pn, 'fp': fp, 'fn': fn, 'tn': tn, 'tp': tp, 'ref_group_value': ref_group}

            row.update(metric_dict)
            row = pd.DataFrame([row])

            bdf = bdf.append(row, ignore_index=True, sort=False)

    return bdf


def audit_fairness(pred_res, protected_attrs, ref_groups, metrics, out_dir, hdf, to_csv=True):
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
    df = pred_res.merge(protected_attrs.reset_index()).drop(id_cols, axis=1).rename(columns={'y_pred': 'score',
                                                                                             'y_true': 'label_value'})

    bias = df.groupby('model_id').apply(compute_bias, ref_groups=ref_groups, metrics=metrics).reset_index(drop=True)
    model_ids = []
    for i in list([i] * 13 for i in range(1,43)):
        model_ids += i

    model_id = pd.Series(model_ids)
    bias['model_id'] = model_id

    hdf.put('pred_bias', bias)
    print('Prediction bias analysis saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'pred_bias.csv')
        bias.to_csv(csv_path, index=False)
        print(f'Prediction bias analysis saved to {csv_path}')


def run(feature_dir, result_dir, model_config):
    """
    Run predictive models configured by the user and save results to the disk

    Parameters
    ----------
    feature_dir : str
        Directory where extracted features are stored

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
            master_table = create_master_table(hdf_in=hdf_feature, hdf_out=hdf_result, out_dir=result_dir,
                                               max_var_miss=model_configs.get('max_var_miss'), label_table_names=['labels'],
                                               standardize=True, by=['course_id'])
            model_info, pred_res = get_pred_res(master_table, features, labels, models, model_configs, 'course_id', out_dir=result_dir, hdf=hdf_result, tune_models=False)
            eval_pred_res(pred_res, metrics, out_dir=result_dir, hdf=hdf_result)
            audit_fairness(pred_res, protected_attrs=master_table['protected_attributes'],
                                           ref_groups=model_configs.get('ref_groups'), metrics=metrics, out_dir=result_dir, hdf=hdf_result)

