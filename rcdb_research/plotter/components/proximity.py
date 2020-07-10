import numpy as np
from pandas.core.common import flatten

import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from typing import Optional, List

from .. import style
from ..utils import configure_axis


def proximity(matrix: np.ndarray,
              clusters: Optional[List[dict]] = None,
              labels: Optional[List] = None,
              distance=True,
              annotate: bool = True,
              title: Optional[str] = 'Proximity matrix',
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
    heatmap_kwargs = {**dict(vmin=0, vmax=1), **(heatmap_kwargs or {})}

    # require labels to be set if clusters is set
    if clusters is not None and labels is None:
        raise ValueError('If clusters param is set, labels param should be set as well.\n')

    labels = np.array(labels if labels is not None else np.arange(matrix.shape[0]), dtype=object)

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    if ax is None:
        side = round(18 / 30. * matrix.shape[0])
        fig, axis = plt.subplots(figsize=(side, side), **fig_kwargs)
    else:
        fig, axis = (plt.gcf(), ax)

    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    cmap = sns.cm.rocket if distance else sns.cm.rocket_r  # noqa

    if clusters is not None:
        cluster_sizes = np.array([len(c['columns']) for c in clusters])

        clustered_labels = np.array(list(flatten([c['columns'] for c in clusters])), dtype=object)
        new_order = np.where(clustered_labels.reshape(clustered_labels.size, 1) == labels)[1]

        sns.heatmap(matrix[np.ix_(new_order, new_order)],
                    xticklabels=labels[new_order],
                    yticklabels=labels[new_order],
                    annot=annotate,
                    cmap=cmap,
                    square=True,
                    cbar_kws={"shrink": .75, 'location': 'bottom'},
                    annot_kws=annotate_kwargs,
                    ax=ax,
                    **heatmap_kwargs)

        left_borders = np.insert(np.cumsum(cluster_sizes), 0, 0)

        for a, b in zip(left_borders, left_borders[1:]):
            axis.add_patch(Rectangle((a, a), b - a, b - a, linewidth=2, fill=False))
    else:
        sns.heatmap(matrix,
                    xticklabels=labels,
                    yticklabels=labels,
                    annot=annotate,
                    cmap=cmap,
                    square=True,
                    cbar_kws={"shrink": .75, 'location': 'bottom'},
                    annot_kws=annotate_kwargs,
                    ax=ax,
                    **heatmap_kwargs)

    if ax is None:
        return fig, axis