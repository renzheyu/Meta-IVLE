import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from src.utils import *

mpl.style.use('ggplot')


def get_best_pred_score(model_info, pred_score, out_dir, hdf, metrics=None, to_csv=True):
    """
    Get the best result for each combination of feature set and label from raw prediction results

    Parameters
    ----------
    model_info : Pandas DataFrame
        model_id | feature | label | model

    pred_score : Pandas DataFrame
        Evaluation score of prediction results, one rwo per model
        Format:
            model_id | metric1 | metric2 | ...

    metrics : list
        List of metrics to find best results by, a subset of all the metrics in pred_score
        If none, use all metrics in pred_score

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    best_pred_score : Pandas DataFrame
        Format:
            feature | label | metric1_model_id | metric1 | metric2_model_id | metric2 | ...
    """
    grand_agg_dict = {
        'acc': ['idxmax', 'max'],
        'fpr': ['idxmin', 'min'],
        'fnr': ['idxmin', 'min']
    }
    agg_dict = {}
    if metrics is not None:
        metric_list = metrics
    else:
        metric_list = pred_score.columns.drop('model_id')
    for metric in metric_list:
        if metric in pred_score.columns.intersection(grand_agg_dict.keys()):
            agg_dict[metric] = grand_agg_dict[metric]
    model_info = categorize_table(model_info)
    best_pred_score = model_info.merge(pred_score).groupby(['feature', 'label']).agg(agg_dict).reset_index()
    best_pred_score.columns = ['_'.join(col) if col[1] != '' else col[0] for col in best_pred_score.columns]
    idx_cols = best_pred_score.columns[best_pred_score.columns.str.contains('idx')]
    for idx_col in idx_cols:
        best_pred_score[idx_col] = model_info['model_id'].loc[best_pred_score[idx_col]].to_numpy()
    best_pred_score.columns = best_pred_score.columns.str.replace(r'idx(min|max)', 'model_id')
    best_pred_score.columns = best_pred_score.columns.str.replace(r'_(min|max)', '')

    hdf.put('best_pred_score', best_pred_score)
    print('Best prediction results saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'best_pred_score.csv')
        best_pred_score.to_csv(csv_path, index=False)
        print(f'Best prediction results saved to {csv_path}')

    return best_pred_score


def get_pred_bias_mat(pred_bias, best_pred_score, out_dir, hdf, neglected_groups=None, small_group_threshold=0.01,
                      to_csv=True):
    """
    Generate a matrix representation of prediction bias against subpopulations, across different features and labels

    Parameters
    ----------
    pred_bias : Pandas DataFrame
        Bias table returned from aequitas.bias
        Format: (excluding irrelevant columns)
            model_col | attribute_name | attribute_value | metric1_disparity | metric1_significance | ...

    best_pred_score : Pandas DataFrame
        Format:
            feature | label | metric1_model_id | metric1 | metric2_model_id | metric2 | ...

    neglected_groups : dict
        Groups to leave out in the plot for each attribute
        Format: {attr1: ['g1_1', 'g1_2'], attr2: ['g2_1']}

    small_group_threshold : float
        Threshold of group size (as proportion of the population size); small groups are left out in the plot

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    pred_bias_mat : Pandas Dataframe
        Matrix representation of prediction biases for each feature set and target against different groups
        Format:
                    Disparity                           Significance
            Target  t1                t2                t1                t2
            Metric  m1       m2       m1       m2       m1       m2       m1       m2
            Group   g1  g2   g1  g2   g1  g2   g1  g2   g1  g2   g1  g2   g1  g2   g1  g2
            Feature
                f1
                f2
    """
    group_cols = ['model_id', 'attribute_name', 'attribute_value']
    f_ref_group = (pred_bias['attribute_value'] == pred_bias['ref_group_value'])
    f_small_group = (pred_bias['group_n'] <= pred_bias['total_n'] * small_group_threshold)
    if neglected_groups is not None:
        f_neglected_group = pred_bias.apply(lambda x: (x['attribute_value'] == neglected_groups.get(x['attribute_name'])),
                                            axis=1)
    else:
        f_neglected_group = 0
    pred_bias = pred_bias[~(f_ref_group | f_small_group | f_neglected_group)]

    pred_bias_mat_long = pd.DataFrame([])
    metrics = best_pred_score.columns[best_pred_score.columns.str.contains('_model_id')].str.replace('_model_id', '')
    bias_metrics = metrics.intersection(pred_bias.columns)
    for bias_metric in bias_metrics:
        model_sub = best_pred_score[['feature', 'label']]
        model_sub['model_id'] = best_pred_score[bias_metric+'_model_id']
        pred_bias_sub = pred_bias[group_cols]
        pred_bias_sub['disparity'] = pred_bias[bias_metric+'_disparity']
        pred_bias_sub['significance'] = pred_bias[bias_metric + '_significance']
        pred_bias_sub['metric'] = bias_metric
        pred_bias_sub = model_sub.merge(pred_bias_sub, how='left')
        pred_bias_mat_long = pred_bias_mat_long.append(pred_bias_sub)
    pred_bias_mat_long = categorize_table(pred_bias_mat_long)

    pred_bias_mat = pd.pivot_table(pred_bias_mat_long, index='feature', values=['disparity', 'significance'],
                                   columns=['label', 'metric', 'attribute_name', 'attribute_value'])

    hdf.put('pred_bias_mat', pred_bias_mat)
    print('Matrix of prediction bias saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'pred_bias_mat.csv')
        pred_bias_mat.to_csv(csv_path, index=True)
        print(f'Matrix of prediction bias saved to {csv_path}')

    return pred_bias_mat


def barh_pred_score(best_pred_score, out_dir):
    """
    Make a horizontal bar plot of prediction results for each metric, where features spread on y-axis and
    labels align side-by-side for each feature

    Parameters
    ----------
    best_pred_score : Pandas DataFrame
        Format:
            feature | label | metric1_model_id | metric1 | metric2_model_id | metric2 | ...

    out_dir : str
        Directory to save the plots

    Returns
    -------
    None
    """
    # TODO: Finish this function
    n_feature = best_pred_score['feature'].nunique()
    n_label = best_pred_score['label'].nunique()
    metrics = best_pred_score.columns[best_pred_score.columns.str.contains('_model_id')].str.replace('_model_id', '')


def heatmap_pred_bias(pred_bias_mat, out_dir, sig_level=0.1):
    """
    Make a heatmap of prediction bias against subpopulations, across different features and labels, based on the
    generated matrix representation

    Parameters
    ----------
    pred_bias_mat : Pandas Dataframe
        Matrix representation of prediction biases for each feature set and target against different groups
        Format:
                    Disparity                           Significance
            Target  t1                t2                t1                t2
            Metric  m1       m2       m1       m2       m1       m2       m1       m2
            Group   g1  g2   g1  g2   g1  g2   g1  g2   g1  g2   g1  g2   g1  g2   g1  g2
            Feature
                f1
                f2

    out_dir : str
        Directory to save the plots

    sig_level : float
        P-value threshold for statistical significance

    """
    def crossout(points, ax, scale=1, **kwargs):
        l = np.array([[[1, 1], [-1, -1]]]) * scale / 2.
        r = np.array([[[-1, 1], [1, -1]]]) * scale / 2.
        p = np.atleast_3d(points).transpose(0, 2, 1)
        c = LineCollection(np.concatenate((l + p, r + p), axis=0), **kwargs)
        ax.add_collection(c)
        return c

    fig, ax = plt.subplots(1, 1, figsize=(40, 25))
    im = ax.matshow(pred_bias_mat['disparity'], cmap='Blues')
    cbar = plt.colorbar(im, shrink=0.4)
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.set_ylabel('Ratio to ref. group', fontsize=22, fontweight='bold')
    ax.set_xticks(np.arange(pred_bias_mat['disparity'].shape[1]))
    ax.set_xticklabels(pred_bias_mat.columns.get_level_values('attribute_value'), rotation=90, fontsize=20,
                       fontweight='bold')
    ax.set_yticks(np.arange(pred_bias_mat['disparity'].shape[0]))
    ax.set_yticklabels(pred_bias_mat['disparity'].index.str.upper(), fontsize=20, fontweight='bold')
    ax.grid(False)
    crossout(np.argwhere(((pred_bias_mat['disparity'].to_numpy() > 1) & (pred_bias_mat['significance'].to_numpy() <
                                                                         sig_level)).T), ax=ax, scale=0.8, color="black")

    png_path = os.path.join(out_dir, 'pred_bias.png')
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    print(f'Plot of prediction bias saved to {png_path}')


def run(result_dir, vis_dir, vis_config):
    """
    Plot prediction results and save to the disk

    Parameters
    ----------
    result_dir : str
        Directory where prediction results are stored

    vis_dir : str
        Directory where visualizations are stored

    vis_config :
        Name of visualization configuration file, with full path

    Returns
    -------
    None
    """
    config = load_yaml(vis_config)

    with pd.HDFStore(os.path.join(result_dir, 'result.h5')) as hdf_result:
        model_info = hdf_result['model_info']
        pred_score = hdf_result['pred_score']
        pred_bias = hdf_result['pred_bias']

        best_pred_score = get_best_pred_score(model_info, pred_score, result_dir, hdf_result)
        pred_bias_mat = get_pred_bias_mat(pred_bias, best_pred_score, result_dir, hdf_result,
                                          neglected_groups=config.get('neglected_groups'),
                                          small_group_threshold=config.get('small_group_threshold'))

        barh_pred_score(best_pred_score, out_dir=vis_dir)
        heatmap_pred_bias(pred_bias_mat, vis_dir)
