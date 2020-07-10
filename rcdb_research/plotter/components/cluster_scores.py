import numpy as np
import pandas as pd
from pandas.core.common import flatten

import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from typing import Optional, List

from .. import style
from ..utils import configure_axis, make_diverging_cmap


def cluster_scores(scores: np.ndarray,
                   clusters: List[dict],
                   labels: List,
                   title: Optional[str] = 'Cluster scores',
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None,
                   fig_kwargs: Optional[dict] = None,
                   ax_kwargs: Optional[dict] = None,
                   annotate_kwargs: Optional[dict] = None,
                   heatmap_kwargs: Optional[dict] = None,
                   ax=None) -> Optional[tuple]:
    fig_kwargs = {
        **style.fig_kwargs(),
        **(fig_kwargs or {})
    }
    _ = fig_kwargs.pop('figsize', None)

    ax_kwargs = {
        **style.ax_kwargs(
            tick_params=dict(bottom=False, labelbottom=False),
        ),
        **(ax_kwargs or {})
    }
    annotate_kwargs = {**dict(), **(annotate_kwargs or {})}
    heatmap_kwargs = {**dict(), **(heatmap_kwargs or {})}

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    if ax is None:
        side = round(18 / 30. * len(labels))
        fig, axis = plt.subplots(figsize=(side, side), **fig_kwargs)
    else:
        fig, axis = (plt.gcf(), ax)

    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    # convert cluster scores into matrix of feature scores
    matrix = np.empty((len(labels), len(labels)))
    matrix[:] = np.nan

    cluster_sizes = [len(c['columns']) for c in clusters]
    cum_cluster_sizes = np.cumsum(cluster_sizes)
    for cid, c in enumerate(clusters):
        for f1id, ft1 in enumerate(c['columns']):
            for f2id, ft2 in enumerate(c['columns']):
                start = cum_cluster_sizes[cid] - cluster_sizes[cid]
                matrix[start + f1id][start + f2id] = scores[cid]

    clustered_labels = list(flatten([c['columns'] for c in clusters]))

    if 'vmin' not in heatmap_kwargs.keys() or 'vmax' not in heatmap_kwargs.keys():
        heatmap_kwargs['vmin'] = max(abs(scores))
        heatmap_kwargs['vmax'] = -heatmap_kwargs['vmin']

    sns.heatmap(matrix,
                xticklabels=clustered_labels,
                yticklabels=clustered_labels,
                cmap=make_diverging_cmap('g', 'w', 'r'),
                square=True,
                linewidths=1,
                linecolor='lightgrey',
                cbar_kws={"shrink": .75, 'location': 'bottom'},
                annot_kws=annotate_kwargs,
                ax=ax,
                **heatmap_kwargs)

    left_borders = np.insert(np.cumsum(cluster_sizes), 0, 0)

    for a, b in zip(left_borders, left_borders[1:]):
        axis.add_patch(Rectangle((a, a), b - a, b - a, linewidth=2, fill=False))

    if ax is None:
        return fig, axis
