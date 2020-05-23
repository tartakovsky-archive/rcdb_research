import matplotlib as mpl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import itertools


def _labels2groups(labels, names=None):
    mapping = {k: [] for k in np.unique(labels)}
    if names is None:
        names = np.arange(len(labels))

    for i in range(len(labels)):
        mapping[labels[i]].append(names[i])
    return mapping


def viz_clusters(matrix, clusters=None, ax=None, true_block_sizes=None, rearrange=False, annotate=True, labels=None,
                 goodbadclusters=None):
    if ax is None:
        side = round(18 / 30. * matrix.shape[0])
        fig, ax = plt.subplots(figsize=(side, side))

    if clusters is None:
        if labels is None:
            labels = np.arange(matrix.shape[0])
        sns.heatmap(matrix, ax=ax, annot=annotate, xticklabels=labels, yticklabels=labels)
        return ax

    if not isinstance(clusters, dict):
        clusters = _labels2groups(clusters)

    if goodbadclusters is not None:
        goodbad = [goodbadclusters[k] for k, _ in sorted(clusters.items(), key=lambda item: sorted(item[1]))]
    clusters = {k: v for k, v in sorted(clusters.items(), key=lambda item: sorted(item[1]))}

    # TODO: use cycler and/or different colormap
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if rearrange:
        new_order = [x for _, y in clusters.items() for x in y]
        sizes = [len(y) for _, y in clusters.items()]

        if labels is None:
            labels = np.arange(sum(sizes))
        sns.heatmap(matrix[np.ix_(new_order, new_order)], xticklabels=labels[new_order], yticklabels=labels[new_order],
                    ax=ax, annot=annotate)

        left_borders = np.insert(np.cumsum(sizes), 0, 0)
        if goodbadclusters is None:
            for a, b in zip(left_borders, left_borders[1:]):
                ax.add_patch(mpl.patches.Rectangle(
                    (a, a), b - a, b - a, fill=False
                ))
        else:  # goodbad is not None
            for a, b, isgood in zip(left_borders, left_borders[1:], goodbadclusters):
                ax.add_patch(mpl.patches.Rectangle(
                    (a, a), b - a, b - a, fill=True, alpha=0.5, hatch='x', color=['red', 'green'][int(isgood)]
                ))
    else:
        if labels is None:
            labels = np.arange(matrix.shape[0])
        sns.heatmap(matrix, ax=ax, annot=annotate, xticklabels=labels, yticklabels=labels)
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

    return ax
