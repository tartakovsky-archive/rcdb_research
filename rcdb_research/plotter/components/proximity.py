import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from typing import Optional, List

from .. import style
from ..utils import configure_axis


def proximity(matrix: np.ndarray,
              labels: Optional[List] = None,
              clusters: Optional[List[list]] = None,
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

    labels = labels if labels is not None else np.arange(matrix.shape[0])

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    if ax is None:
        side = round(18 / 30. * matrix.shape[0])
        fig, axis = plt.subplots(figsize=(side, side), **fig_kwargs)
    else:
        fig, axis = (plt.gcf(), ax)

    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    mask = np.triu(np.ones(matrix.shape), k=1).astype(np.bool)

    #     axis.yaxis.set_label_position('right')
    #     axis.yaxis.set_ticks_position('right')
    cmap = sns.cm.rocket if distance else sns.cm.rocket_r
    sns.heatmap(matrix, mask=mask, ax=ax, yticklabels=labels, annot=annotate, cmap=cmap, square=True,
                cbar_kws={"shrink": .75, 'location': 'bottom'}, annot_kws=annotate_kwargs, **heatmap_kwargs)

    if ax is None:
        return fig, axis
