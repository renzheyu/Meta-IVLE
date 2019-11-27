import os
import numpy as np
import pandas as pd
from functools import reduce
from itertools import product
from src.utils import *

from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import robust_scale
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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


def get_model(model_name, params=None):
    """
    Configure a classifier instance given the input model name
    Currently no hyperparameter tuning but can accommodate in future versions

    Parameters
    ----------
    model_name : str

    params : dict or None
        If not None, a dictionary of parameters for the specified classifier, in the form of {param_name: param_value}

    Returns
    -------
    clf : Scikit-learn estimator
        Configured classifier
    """
    clf_dict = {
        'logistic_regression': LogisticRegression(),
        'svm': SVC(),
        'random_forest': RandomForestClassifier()
    }
    clf = clf_dict[model_name]
    if params is not None:
        clf.set_params(**params)
    return clf


def get_pred_res(master_table, features, labels, models, group_var, rseed, out_dir, hdf, model_col='model_id',
                 to_csv=True):
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

    group_var : str
        Variable name to group samples for cross validation. Currently must be in the index of master_table

    rseed : int
        Pseudo-random number

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    model_col : string
        Resulting column name that identifies unique models

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
    pred_res = pd.DataFrame([], columns=[''])
    unique_id = master_table.index.to_frame().reset_index(drop=True)
    pred_res = pd.DataFrame([], columns=unique_id.columns.tolist()+['model_id', 'y_true', 'y_pred'])
    for (feature, label) in product(features, labels):
        X = get_features(master_table, feature).to_numpy()
        y = get_labels(master_table, label).to_numpy()
        groups = master_table.index.get_level_values(group_var)
        for model in models:
            print(f'Predicting {label} using {feature} via {model}...')
            clf = get_model(model, params={'random_state': rseed})
            logo = LeaveOneGroupOut()
            estimator = make_pipeline(make_union(SimpleImputer(strategy='constant', fill_value=0), MissingIndicator(
                features='all')), clf)
            predicted = cross_val_predict(estimator, X, y, groups=groups, cv=logo)
            model_id = len(model_info) + 1
            model_info = model_info.append({'model_id': model_id, 'feature': feature, 'label': label, 'model': model},
                              ignore_index=True)
            res = pd.DataFrame({model_col: model_id, 'y_true': y, 'y_pred': predicted})
            pred_res = pred_res.append(pd.concat([unique_id, res], axis=1))

    hdf.put('model_info', model_info)
    print('Model info saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'model_info.csv')
        model_info.to_csv(csv_path, index=False)
        print(f'Model info saved to {csv_path}')
    return model_info, pred_res


def eval_pred_res(pred_res, metrics, out_dir, hdf, model_col='model_id', comp_model=False, to_csv=True):
    """
    Evaluate raw prediction results based on given metrics

    Parameters
    ----------
    pred_res : Pandas DataFrame
        Raw prediction results, one row per model per entity
        Format:
            entity_id | model_col | y_true | y_pred

    metrics : list
        List of metric names

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    model_col : string
        Column name in pred_res that identifies unique models

    comp_model : boolean
        If true, compare each pair of models and generate and return a comparison table

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Results
    -------
    pred_eval_hits : Pandas DataFrame
        Raw prediction results concatenated with binary evaluation of these results given each metric
        For each metric, 1 assigned to predictions contributing to the numerator of its calculation, 0 to those
        contributing to the denominator, and NaN to those not included in the calculation (Ex. false positive rate =
        false positives / true negatives)
        Format:
            entity_id | model_col | metric1_ind | metric2_ind | ...

    pred_eval_score : Pandas DataFrame
        Evaluation score of prediction results, one rwo per model
        Format:
            model_col | metric1 | metric2 | ...

    model_comp : Pandas DataFrame (currently not used)
        Results of statistical tests of pairwise model result comparisons
        Format:
            model_id_1 | model_id_2 | diff_metric1 | p_val_metrics1 | ...
    """
    pred_eval_hits = pred_res.drop(['y_true', 'y_pred'], axis=1)
    for metric in metrics:
        print(f'Calculating {metric}...')
        pred_eval_hits[metric] = get_pred_eval_hits(metric, pred_res['y_true'], pred_res['y_pred'])
    pred_eval_score = pred_eval_hits.groupby(model_col)[metrics].mean().reset_index()
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


def audit_fairness(pred_res, protected_attrs, ref_groups, out_dir, hdf, model_col='model_id', to_csv=True):
    """
    Evaluate raw prediction results based on given metrics

    Parameters
    ----------
    pred_res : Pandas DataFrame
        Raw prediction results, one row per model per entity
        Ex:
            entity_id | model_col | y_true | y_pred

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

    model_col : string
        Column name in pred_res that identifies unique models

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Results
    -------
    bias : Pandas DataFrame
        Bias measures against different groups as calculated by aequitas.bias
        Format:
            model_col | attribute_name | attribute_value | bias1_disparity | bias1_significance | ...
    """
    def compute_bias(df, ref_groups):
        g = Group()
        b = Bias()
        xtab, _ = g.get_crosstabs(df)
        bdf = b.get_disparity_predefined_groups(xtab, original_df=df, ref_groups_dict=ref_groups,
                                                check_significance=True, mask_significance=False)
        return bdf

    id_cols = protected_attrs.index.to_frame().columns
    df = pred_res.merge(protected_attrs.reset_index()).drop(id_cols, axis=1).rename(columns={'y_pred': 'score',
                                                                                             'y_true': 'label_value'})
    bias = df.groupby(model_col).apply(compute_bias, (ref_groups)).reset_index(drop=True)

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

    config = load_yaml(model_config)

    features = config.get('features')
    labels = config.get('labels')
    models = config.get('models')
    metrics = config.get('metrics')

    with pd.HDFStore(os.path.join(feature_dir, 'feature.h5')) as hdf_feature:
        with pd.HDFStore(os.path.join(result_dir, 'result.h5')) as hdf_result:
            master_table = create_master_table(hdf_in=hdf_feature, hdf_out=hdf_result, out_dir=result_dir,
                                               max_var_miss=config.get('max_var_miss'), label_table_names=['labels'],
                                               standardize=True, by=['course_id'])
            model_info, pred_res = get_pred_res(master_table, features, labels, models, 'course_id', rseed=config.get(
                'random_seed'), out_dir=result_dir, hdf=hdf_result)
            eval_pred_res(pred_res, metrics, out_dir=result_dir, hdf=hdf_result)
            audit_fairness(pred_res, protected_attrs=master_table['protected_attributes'],
                                           ref_groups=config.get('ref_groups'), out_dir=result_dir, hdf=hdf_result)
