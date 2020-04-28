import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from src.utils import *

mpl.style.use('ggplot')


def get_best_pred_score(model_info, pred_score, out_dir, hdf, metrics=None, to_hdf=True, to_csv=True):
    """
    Get the best result for each combination of feature set and label from raw prediction results

    Parameters
    ----------
    model_info : Pandas DataFrame
        model_id | feature | label | model

    pred_score : Pandas DataFrame
        Evaluation score of prediction results, one row per model
        Format:
            model_id | tp | tn | fp | fn | f1 | metric1 | metric2 | ...

    metrics : list
        List of metrics to find best results by, a subset of all the metrics in pred_score
        If none, use all metrics in pred_score

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_hdf : boolean
        Whether to save the resulting table in the specified HDFStore object

    to_csv : boolean
        Whether to save the resulting table in a .csv file (in addition to HDF) for easier examination

    Returns
    -------
    best_pred_score : Pandas DataFrame
        Format:
            feature | label | model_id | metric1 | metric2 | ...
    """
    if metrics is not None:
        metric_list = metrics
    else:
        metric_list = pred_score.columns.drop(['model_id', 'tp', 'tn', 'fp', 'fn'])
    model_info = categorize_table(model_info)
    best_pred_score = model_info.merge(pred_score).groupby(['feature', 'label'])['f1'].idxmax().reset_index().rename(
        {'f1': 'model_id'}, axis=1)
    best_pred_score['model_id'] = best_pred_score['model_id'] + 1
    best_pred_score = best_pred_score.merge(pred_score[['model_id']+metric_list])

    if to_hdf:
        hdf.put('best_pred_score', best_pred_score)
        print('Best prediction results saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'best_pred_score.csv')
        best_pred_score.to_csv(csv_path, index=False)
        print(f'Best prediction results saved to {csv_path}')

    return best_pred_score


def get_pred_bias_mat(pred_bias, best_pred_score, out_dir, hdf, neglected_groups=None, small_group_threshold=0.01,
                      to_hdf=True, to_csv=True):
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
            feature | label | model_id | metric1 | metric2_model_id | metric2 | ...

    neglected_groups : dict
        Groups to leave out in the plot for each attribute
        Format: {attr1: ['g1_1', 'g1_2'], attr2: ['g2_1']}

    small_group_threshold : float
        Threshold of group size (as proportion of the population size); small groups are left out in the plot

    out_dir : str
        Directory to save the resulting table

    hdf : HDFStore object
        Where the resulting table is stored (comparable to a schema in databases)

    to_hdf : boolean
        Whether to save the resulting table in the specified HDFStore object

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
    f_ref_group = (pred_bias['attribute_value'] == pred_bias['ref_group_value'])
    f_small_group = (pred_bias['group_n'] <= pred_bias['total_n'] * small_group_threshold)
    if neglected_groups is not None:
        f_neglected_group = pred_bias.apply(is_neglected_group, axis=1, args=(neglected_groups,))
    else:
        f_neglected_group = False
    pred_bias_valid = pred_bias[~(f_ref_group | f_small_group | f_neglected_group)]

    pred_bias_mat_long = pd.DataFrame([])
    metrics = best_pred_score.columns.drop(['feature', 'label', 'model_id'])
    bias_metrics = metrics.intersection(pred_bias_valid.columns)
    for bias_metric in bias_metrics:
        model_sub = best_pred_score[['feature', 'label']]
        model_sub['model_id'] = best_pred_score['model_id']
        pred_bias_sub = pred_bias_valid[['model_id', 'attribute_name', 'attribute_value']]
        pred_bias_sub['disparity'] = pred_bias_valid[bias_metric + '_disparity']
        pred_bias_sub['significance'] = pred_bias_valid[bias_metric + '_significance']
        pred_bias_sub['metric'] = bias_metric
        pred_bias_sub = model_sub.merge(pred_bias_sub, how='left')
        pred_bias_mat_long = pred_bias_mat_long.append(pred_bias_sub)
    pred_bias_mat_long = categorize_table(pred_bias_mat_long)

    pred_bias_mat = pd.pivot_table(pred_bias_mat_long, index='feature', values=['disparity', 'significance'],
                                   columns=['label', 'metric', 'attribute_name', 'attribute_value'])

    if to_hdf:
        hdf.put('pred_bias_mat', pred_bias_mat)
        print('Matrix of prediction bias saved to HDFStore')
    if to_csv:
        csv_path = os.path.join(out_dir, 'pred_bias_mat.csv')
        pred_bias_mat.to_csv(csv_path, index=True)
        print(f'Matrix of prediction bias saved to {csv_path}')

    return pred_bias_mat

def bar_target_dist(pred_bias, model_info, out_dir, neglected_groups=None):
    # TODO: docstring
    target_dist = model_info.drop_duplicates('label')[['label', 'model_id']]
    target_dist = target_dist.merge(pred_bias, on='model_id', how='left')
    if neglected_groups is not None:
        f_neglected_group = target_dist.apply(is_neglected_group, axis=1, args=(neglected_groups,))
        target_dist = target_dist[~f_neglected_group]

    targets = target_dist['label'].unique()
    n_ticks = target_dist['attribute_value'].nunique() + target_dist['attribute_name'].nunique()
    cls_names = ['label_pos', 'label_neg']
    cls_labs = {'label_pos': 'upper half', 'label_neg': 'lower half'}

    tick_names = []
    tick_names_flag = False
    for tar in targets:
        fix, ax = plt.subplots(figsize=(5, n_ticks * 0.5))
        width = 0.75 / len(cls_names)
        tar_dist = target_dist.query(f'label == "{tar}"')
        for i, cls in enumerate(cls_names):
            cls_vals = []
            for attr in tar_dist['attribute_name'].unique():
                if not tick_names_flag:
                    tick_names.append(attr.upper())
                    tick_names += list(tar_dist.query(f'attribute_name == "{attr}"')['attribute_value'])
                cls_vals.append(0)
                cls_vals += list(tar_dist.query(f'attribute_name == "{attr}"')[cls])
            tick_names_flag = True
            ax.barh(np.arange(n_ticks)+i*width, cls_vals, width, left=0)
        if cls_labs is not None:
            cls_legend = [cls_labs.get(cls) for cls in cls_names]
        else:
            cls_legend = cls_names
        ax.legend(cls_legend, fontsize=12, loc='upper right')
        ax.set_xlabel('# students'.upper(), fontsize=13)
        ax.set_yticks(np.arange(n_ticks) + width / 2)
        ax.set_yticklabels(tick_names, fontsize=10)
        ax.set_ylim(0, n_ticks-width)
        ax.invert_yaxis()
        ax.grid(False, axis='y')
        ax.tick_params(labelsize=14, top='off', left='off')

        png_path = os.path.join(out_dir, f'target_dist_{tar}.png')
        plt.savefig(png_path, dpi=400, bbox_inches='tight')
        print(f'Plot of target distribution saved to {png_path}')


# def bar_target_dist(pred_bias, model_info, out_dir, neglected_groups=None, baseline=None, alias=None):
#     # TODO: docstring
#     target_dist = model_info.drop_duplicates('label')[['label', 'model_id']]
#     target_dist = target_dist.merge(pred_bias, on='model_id', how='left')
#     if neglected_groups is not None:
#         f_neglected_group = target_dist.apply(is_neglected_group, axis=1, args=(neglected_groups,))
#         target_dist = target_dist[~f_neglected_group]
#
#     targets = target_dist['label'].unique()
#     n_ticks = target_dist['attribute_value'].nunique() + target_dist['attribute_name'].nunique()
#     fix, ax = plt.subplots(figsize=(5, n_ticks*0.5))
#     width = 0.75 / len(targets)
#
#     tick_names = []
#     tick_names_flag = False
#     for (i, tar) in enumerate(targets):
#         tar_vals = []
#         tar_dist = target_dist.query(f'label == "{tar}"')
#         for attr in tar_dist['attribute_name'].unique():
#             if not tick_names_flag:
#                 tick_names.append(attr.upper())
#                 tick_names += list(tar_dist.query(f'attribute_name == "{attr}"')['attribute_value'])
#             tar_vals.append(0)
#             tar_vals += list(tar_dist.query(f'attribute_name == "{attr}"')['label_pos'] / tar_dist.query(
#                 f'attribute_name == "{attr}"')['group_n'] * 100)
#         tick_names_flag = True
#         ax.barh(np.arange(n_ticks)+i*width, tar_vals, width, left=0)
#     if alias is not None:
#         target_names = [alias.get(tar) for tar in targets]
#     else:
#         target_names = targets
#     ax.legend(target_names, fontsize=12, loc='upper right')
#     if baseline is not None:
#         ax.vlines(baseline*100, 0, n_ticks-width, linestyles='dashed')
#     ax.set_xlabel('% in upper half of class'.upper(), fontsize=13)
#     ax.set_yticks(np.arange(n_ticks) + width / 2)
#     ax.set_yticklabels(tick_names, fontsize=10)
#     ax.set_ylim(0, n_ticks-width)
#     ax.invert_yaxis()
#     ax.grid(False, axis='y')
#     ax.tick_params(labelsize=14, top='off', left='off')
#
#     png_path = os.path.join(out_dir, f'target_dist.png')
#     plt.savefig(png_path, dpi=400, bbox_inches='tight')
#     print(f'Plot of target distribution saved to {png_path}')


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
        Format: (Column level names are not exact)
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

    for label in pred_bias_mat.columns.get_level_values('label').unique():
        disparity_mat = pred_bias_mat.loc[:, ('disparity', label)]
        significance_mat = pred_bias_mat.loc[:, ('significance', label)]
        fig, ax = plt.subplots(1, 1, figsize=(25, 25))
        im = ax.matshow(disparity_mat, cmap='Blues')
        ax.set_xticks(np.arange(disparity_mat.shape[1]))
        ax.set_xticklabels(disparity_mat.columns.get_level_values('attribute_value'), rotation=90, fontsize=20,
                           fontweight='bold')
        ax.set_yticks(np.arange(disparity_mat.shape[0]))
        ax.set_yticklabels(disparity_mat.index.str.upper(), fontsize=20, fontweight='bold')
        ax.grid(False)
        crossout(np.argwhere((significance_mat.to_numpy() < sig_level).T), ax=ax, scale=0.8, color="black")
        cbar = plt.colorbar(im, shrink=0.3, ticks=np.arange(0, 10, 2), ax=ax)
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.set_ylabel('Ratio to ref. group', fontsize=22, fontweight='bold')

        png_path = os.path.join(out_dir, f'pred_bias_{label}.png')
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

        # best_pred_score = get_best_pred_score(model_info, pred_score, result_dir, hdf_result, metrics=config.get(
        #     'metrics'))
        best_pred_score = hdf_result['best_pred_score']
        # pred_bias_mat = get_pred_bias_mat(pred_bias, best_pred_score, result_dir, hdf_result,
        #                                   neglected_groups=config.get('neglected_groups'),
        #                                   small_group_threshold=config.get('small_group_threshold'))
        # pred_bias_mat = hdf_result['pred_bias_mat']

        # barh_pred_score(best_pred_score, out_dir=vis_dir)
        # heatmap_pred_bias(pred_bias_mat, vis_dir, sig_level=config.get('sig_level'))
        bar_target_dist(pred_bias, model_info, vis_dir, neglected_groups=config.get('neglected_groups'))