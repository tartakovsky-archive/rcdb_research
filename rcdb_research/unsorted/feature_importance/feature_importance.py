# flake8: noqa

import pandas as pd
import numpy as np
from rcdb_research.sampling.cv import CombinatorialCV, split_indexes_to_bars, predict_splits, predicts_to_paths

from rcdb_research.sampling.bootstrap import bootstrap, bootstrap_2d, optimal_block_size
from rcdb_research.sampling.bootstrap import bootstrap_path, bootstrap_path_2d

from rcdb_research.scoring import score_2d, score_3d
from rcdb_research.scoring import score_path, score_path_2d, score_path_3d

from tqdm.auto import tqdm


def compute_MDA_evenbetter(X, y, clf, agglomeration, score, other_params):
    features = agglomeration.fit_transform(X)
    features = pd.DataFrame(features, index=X.index)

    labels = agglomeration.labels_
    matrix = agglomeration.affinity(X.T)

    cv = CombinatorialCV(n_folds=other_params['N'], k_tests=other_params['k'], embargo=other_params['embargo'],
                         tainted_up_to=None)
    indexes = cv.split(features)
    splits = split_indexes_to_bars(features, y, indexes)
    rs = np.random.RandomState(other_params['random_seed'])

    for split in splits:
        split['X_test'] = split['X_test'].copy()

    scores = {ft: [] for ft in splits[0]['X_test'].columns}
    for split in tqdm(splits, desc='splits processed'):
        clf.fit(split['X_train'], split['y_train'])
        y_pred = clf.predict_proba(split['X_test'])[:, 1]
        baseline_sc = score(split['y_test'], y_pred)
        for ft in split['X_test'].columns:
            split['original'] = split['X_test'][ft].copy()
            for _ in range(other_params['n_shuffles']):
                new_order = rs.choice(split['X_test'].index.size, size=split['X_test'].index.size, replace=False)
                split['X_test'][ft] = split['X_test'][ft].values[new_order]

                y_pred = clf.predict_proba(split['X_test'])[:, 1]
                sc = score(split['y_test'], y_pred)
                scores[ft].append((baseline_sc - sc))
            split['X_test'][ft] = split['original'].copy()

    decreases = []
    for ft, sc in scores.items():
        mdecrease = np.median(sc)
        decreases.append({
            'cluster': ft,
            'decrease': mdecrease,
            'lower_bound': mdecrease - np.percentile(sc, 5),
            'upper_bound': np.percentile(sc, 100 - 5) - mdecrease,
        })
    decreases = pd.DataFrame.from_records(decreases)

    return matrix, labels, decreases, features


#######################################################################################
import matplotlib as mpl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import minmax_scale


def _labels2groups(labels, names=None):
    mapping = {k: [] for k in np.unique(labels)}
    if names is None:
        names = np.arange(len(labels))

    for i in range(len(labels)):
        mapping[labels[i]].append(names[i])
    return mapping


def viz_clusters(matrix, clusters=None, fig=None, ax=None, true_block_sizes=None, rearrange=True, annotate=True,
                 labels=None,
                 goodbadclusters=None):
    if ax is None:
        side = round(18 / 30. * matrix.shape[0])
        fig, ax = plt.subplots(figsize=(side, side))

    if clusters is None:
        if labels is None:
            labels = np.arange(matrix.shape[0])
        sns.heatmap(matrix, ax=ax, annot=annotate, xticklabels=labels, yticklabels=labels, square=True,
                    cbar_kws={"shrink": .75})
        ax.set_title('distance')
        return fig, ax

    if not isinstance(clusters, dict):
        clusters = _labels2groups(clusters)

    if goodbadclusters is not None:
        goodbadclusters = pd.Series(
            [goodbadclusters[k] for k, _ in sorted(clusters.items(), key=lambda item: sorted(item[1]))])
    clusters = {k: v for k, v in sorted(clusters.items(), key=lambda item: sorted(item[1]))}

    # TODO: use cycler and/or different colormap
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if rearrange:
        new_order = [x for _, y in clusters.items() for x in y]
        sizes = [len(y) for _, y in clusters.items()]

        if labels is None:
            labels = np.arange(sum(sizes))
        sns.heatmap(matrix[np.ix_(new_order, new_order)], xticklabels=labels[new_order], yticklabels=labels[new_order],
                    ax=ax, annot=annotate, square=True, cbar_kws={"shrink": .75})

        left_borders = np.insert(np.cumsum(sizes), 0, 0)
        if goodbadclusters is None:
            for a, b in zip(left_borders, left_borders[1:]):
                ax.add_patch(mpl.patches.Rectangle(
                    (a, a), b - a, b - a, fill=False
                ))
        else:  # goodbadclusters is not None
            for a, b, goodness in zip(left_borders, left_borders[1:], goodbadclusters):
                color = ['red', 'green'][int(goodness > 0)]
                ax.add_patch(mpl.patches.Rectangle(
                    (a, a), b - a, b - a, fill=True, alpha=0.5, hatch='x', color=color
                ))
    else:
        if labels is None:
            labels = np.arange(matrix.shape[0])
        sns.heatmap(matrix, ax=ax, annot=annotate, xticklabels=labels, yticklabels=labels, square=True,
                    cbar_kws={"shrink": .75})
        for color, (cluster, fts) in zip(colors, clusters.items()):
            for x, y in np.array(list(itertools.product(fts, fts))) + 0.45:
                ax.scatter(x, y, color='black', linewidths=5)
                ax.scatter(x, y, color=color)

    if true_block_sizes is not None:
        left_borders = np.insert(np.cumsum(true_block_sizes), 0, 0)
        for a, b in zip(left_borders, left_borders[1:]):
            ax.add_patch(mpl.patches.Rectangle(
                (a, a), b - a, b - a, fill=False
            ))

    ax.set_title('distance & clusters')

    return fig, ax


#######################################################################################
from natsort import natsorted


def viz_decrease_matrix(groups, decreases, ax=None):
    if ax is None:
        side = round(18 / 30. * matrix.shape[0])
        fig, ax = plt.subplots(figsize=(side, side))
    decreases = decreases.set_index('cluster')
    groups = {k: v for k, v in natsorted(groups.items(), key=lambda item: natsorted(item[1]))}
    new_order = [x for _, y in groups.items() for x in y]
    sizes = [len(y) for _, y in groups.items()]
    rows = []
    for k, v in groups.items():
        for ft1 in v:
            for ft2 in v:
                rows.append((ft1, ft2, decreases.loc[k]['decrease']))
    result = pd.DataFrame.from_records(rows, columns=['ft1', 'ft2', 'decrease'])
    vmax = np.abs(result['decrease'].values).max()
    vmin = -vmax
    table = result.pivot(index='ft1', columns='ft2', values='decrease')
    sns.heatmap(table.loc[new_order, new_order], annot=False, cmap='RdYlGn', vmin=vmin, vmax=vmax, linewidths=1,
                linecolor='lightgrey', ax=ax, square=True, cbar_kws={"shrink": .75})
    ax.set(xlabel=None, ylabel=None)
    ax.set(title='MDA score')


#######################################################################################
import matplotlib.ticker as mtick


def viz_decrease(decreases, groups=None, ax=None, show_std=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(18, round(6 / 10 * len(decreases))))

    def special_format(lst):
        return f'{lst[0]} + {len(lst) - 1}'

    #     decreases = decreases.iloc[np.argsort((decreases['decrease'] - decreases['lower_bound']))]
    #     decreases = decreases.iloc[np.argsort((decreases['decrease']))]

    if show_std:
        colors = np.where(decreases['decrease'] - decreases['lower_bound'] > 0, 'C0', 'C3')
        ax.barh(np.arange(decreases.shape[0]), decreases['decrease'],
                xerr=decreases[['lower_bound', 'upper_bound']].T.values, color=colors)
    else:
        colors = np.where(decreases['decrease'] > 0, 'C0', 'C3')
        ax.barh(np.arange(decreases.shape[0]), decreases['decrease'], color=colors)

    #     ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_yticks(np.arange(decreases.shape[0]))
    if groups is None:
        ax.set_yticklabels(decreases['cluster'])
    else:
        ax.set_yticklabels([special_format(groups[k]) for k in decreases['cluster']])
    ax.set(title='MDA score')
    ax.axvline(0, color='black')
    ax.grid()


#######################################################################################
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram


def matrix2condensed(matrix):
    res = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            res.append(matrix[i][j])
    return np.array(res)


def hierarchial_clustering_linkage(matrix, method):
    matrix = np.clip(matrix, 0, 1)
    L = sch.linkage(matrix2condensed(matrix), method=method)
    return L


def viz_dendrogram(matrix, labels, threshold, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 10))

    L = hierarchial_clustering_linkage(matrix, 'complete')
    dendrogram(L, labels=labels, ax=ax, leaf_rotation=0, orientation='right', color_threshold=threshold)
    ax.axvline(threshold, color='grey', linestyle='--', label='threshold')
    ax.set_title('dendrogram')
    ax.locator_params(nbins=40, axis='x')
    ax.legend(loc='upper right')
    return ax


#######################################################################################
def dump_decrease(decreases, groups, score):
    #     decreases = decreases.iloc[np.argsort(-1 * (decreases['decrease'] - decreases['lower_bound']))]
    decreases = decreases.iloc[np.argsort(-1 * (decreases['decrease']))]
    for _, row in decreases.iterrows():
        cluster = row['cluster']
        decrease = row['decrease']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']
        print('=' * 80)
        print(
            f'#{int(cluster)}, score: {decrease:.3f} ({decrease - lower_bound:.3f}, {decrease + upper_bound:.3f}) ({score})')
        print('-' * 80)
        for ft in groups[cluster]:
            print(' ', ft)


#######################################################################################
def extract_above_threshold(X, decreases, thr=0):
    return X[decreases[decreases['decrease'] - decreases['lower_bound'] > thr]['cluster']]


#######################################################################################
import sklearn.preprocessing


def build_transformer(clusters, groups, decreases, thr=0):
    def special_format(lst):
        return f'{lst[0]} + {len(lst) - 1}' if len(lst) > 1 else f'{lst[0]}'

    def fn(X):
        X = X.groupby(axis='columns', by=clusters).agg('mean')
        X.columns = [special_format(groups[k]) for k in decreases['cluster']]
        return X.iloc[:, decreases[decreases['decrease'] - decreases['lower_bound'] > thr]['cluster']]

    return sklearn.preprocessing.FunctionTransformer(fn)


def build_transformer2(agglomeration, clusters, decreases, thr=0):
    def fn(X):
        X = pd.DataFrame(agglomeration.transform(X))
        return X[decreases[decreases['decrease'] - decreases['lower_bound'] > thr]['cluster']]

    return sklearn.preprocessing.FunctionTransformer(fn)


#######################################################################################
from tqdm.auto import tqdm


def compute_MDA_refactored(X, y, clf, agglomeration, score, other_params, score_path, raw):
    features = agglomeration.fit_transform(X)
    features = pd.DataFrame(features, index=X.index)

    labels = agglomeration.labels_
    matrix = agglomeration.affinity(X.T)

    cv = CombinatorialCV(n_folds=other_params['N'], k_tests=other_params['k'], embargo=other_params['embargo'],
                         tainted_up_to=None)
    indexes = cv.split(features)
    splits = split_indexes_to_bars(features, y, indexes)
    rs = np.random.RandomState(other_params['random_seed'])

    for split in splits:
        split['X_test'] = split['X_test'].copy()

    ft_names = splits[0]['X_test'].columns

    scores = {ft: [] for ft in ft_names}

    predicted_splits = {ft: [[] for _ in range(other_params['n_shuffles'])] for ft in ft_names}
    baselines = []

    for split in tqdm(splits, desc='splits processed'):
        clf.fit(split['X_train'], split['y_train'])
        y_pred = clf.predict_proba(split['X_test'])[:, 1]
        baselines.append({
            'y_true': split['y_test'],
            'y_pred': y_pred,
            'index': split['y_test'].index
        })
        for ft in split['X_test'].columns:
            split['original'] = split['X_test'][ft].copy()
            for shf in range(other_params['n_shuffles']):
                rs.shuffle(split['X_test'][ft].values)
                y_pred = clf.predict_proba(split['X_test'])[:, 1]
                predicted_splits[ft][shf].append({
                    'y_true': split['y_test'],
                    'y_pred': y_pred,
                    'index': split['y_test'].index
                })
            split['X_test'][ft] = split['original'].copy()

    N = other_params['N']
    k = other_params['k']
    n_shuffles = other_params['n_shuffles']

    if score_path:
        predicted_paths = {
            ft: [predicts_to_paths(predicted_splits[ft][shf], n_folds=N, k_tests=k) for shf in range(n_shuffles)]
            for ft in ft_names
        }
        scores = {
            ft: np.array(score_path_3d(predicted_paths[ft], score)).ravel() for ft in ft_names
        }
        predicted_baseline_paths = predicts_to_paths(baselines, n_folds=N, k_tests=k)
        baseline_scores = np.repeat(np.array(score_path_2d(predicted_baseline_paths, score)).ravel(), n_shuffles)
    else:
        scores = {
            ft: np.array(score_path_3d(predicted_splits[ft], score)).ravel() for ft in ft_names
        }
        baseline_scores = np.repeat(np.array(score_path_2d(baselines, score)).ravel(), n_shuffles)

    if raw:
        return scores, baseline_scores

    decreases = []
    for ft in ft_names:
        sc = baseline_scores - scores[ft]
        mdecrease = np.median(sc)
        q025 = np.percentile(sc, 2.5)
        q975 = np.percentile(sc, 97.5)
        decreases.append({
            'cluster': ft,
            'decrease': mdecrease,
            'lower_bound': mdecrease - q025,
            'upper_bound': q975 - mdecrease,
        })
    decreases = pd.DataFrame.from_records(decreases)

    return matrix, labels, decreases, features


#######################################################################################
from matplotlib.gridspec import GridSpec
from natsort import natsorted


def perform_mda(X, y, clf, agglomeration, score, other_params, score_path=False, threshold_pct=0, silent=False):
    X = X[natsorted(X.columns)]
    #     matrix, clusters, decreases, features = compute_MDA_evenbetter(X, y, clf, agglomeration, score, other_params)
    #     matrix, clusters, decreases, features = compute_MDA_eli5(X, y, clf, agglomeration, score, other_params)
    #     matrix, clusters, decreases, features = compute_MDA_mlfinlab(X, y, clf, agglomeration, score, other_params)
    matrix, clusters, decreases, features = compute_MDA_refactored(X, y, clf, agglomeration, score, other_params,
                                                                   score_path, raw=False)
    groups = _labels2groups(clusters, names=X.columns)
    # retval = extract_above_threshold(features, decreases, threshold_pct)
    retval = build_transformer(clusters, groups, decreases)
    # retval = build_transformer2(agglomeration, clusters, decreases)

    if silent:
        return retval

    square_side = max(16, round(18 / 30 * X.shape[1]))
    bars_height = round(6 / 10 * len(decreases))

    fig = plt.figure(constrained_layout=True, figsize=(square_side * 2, square_side + bars_height + square_side))
    gs = GridSpec(3, 4, height_ratios=[square_side, bars_height, square_side], figure=fig)
    axes = [
        fig.add_subplot(gs[0, :2]),
        fig.add_subplot(gs[0, 2:]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[2, 1:3]),
    ]

    viz_clusters(matrix, clusters, rearrange=True, labels=X.columns, ax=axes[0])
    viz_decrease_matrix(groups, decreases, ax=axes[1])
    viz_decrease(decreases.sort_values('decrease'), groups, ax=axes[2])
    viz_dendrogram(matrix, X.columns, agglomeration.distance_threshold, ax=axes[3])

    fig.suptitle('MDA report ({})'.format(score.__name__), fontsize=18, y=0.975)
    display(fig)
    plt.close(fig)

    dump_decrease(decreases, groups, score.__name__)

    return retval


#######################################################################################
import warnings
from tqdm.auto import tqdm
from natsort import natsorted

from rcdb_research.metrics.proximity import nid
from sklearn.metrics import pairwise_distances
from sklearn.cluster import FeatureAgglomeration


def perform_sfi(X, y, clf, agglomeration, score, other_params, shifted, score_path=False):
    results = {}

    def special_format(lst):
        return f'{lst[0]} + {len(lst) - 1}'

    if not shifted:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            features = pd.DataFrame(agglomeration.fit_transform(X), index=X.index)
            features.columns = list(map(special_format, _labels2groups(agglomeration.labels_, X.columns).values()))
    else:
        if agglomeration is None:
            agglomeration = FeatureAgglomeration(
                n_clusters=None,
                affinity=lambda X: pairwise_distances(X, metric=nid),
                linkage='complete',
                distance_threshold=0.75,
            )
        features = X.shift(-1)[:-1]
        y = y[:-1]

    fig, ax = plt.subplots(figsize=(20, max(16, round(6 / 10 * features.shape[1]))))

    for ft in tqdm(features.columns, desc='features processed'):
        results[ft] = compute_MDA_refactored(
            features[[ft, ft]], y, clf, agglomeration, score,
            other_params=other_params,
            score_path=score_path,
            raw=False
        )[2]

    fts, decreases = list(zip(*results.items()))
    groups = {idx: [ft] for idx, ft in enumerate(fts)}
    decreases = pd.concat(decreases)
    decreases['cluster'] = np.arange(decreases.shape[0])
    decreases = decreases.reset_index(drop=True)

    if shifted:
        decreases['clustername'] = features.columns
        decreases = decreases.loc[list(reversed(natsorted(decreases.index, lambda x: decreases.loc[x]['clustername'])))]
        decreases = decreases.drop(columns='clustername')
    else:
        decreases = decreases.sort_values('decrease')

    viz_decrease(decreases, groups, ax=ax)
    ax.set_title('SFI ({}) ({})'.format(score.__name__, ['non shifted', 'shifted'][int(shifted)]))

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
