import numpy as np
from pandas.core.common import flatten

from typing import Optional, List

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from .. import style
from ..utils import configure_axis

from .. import primitives as prim
from .. import components as comp


def clustered_importance(proximity_matrix,
                         scores,
                         clusters,
                         labels,
                         fig_kwargs: Optional[dict] = None):
    fig_kwargs = {
        **style.fig_kwargs(),
        **(fig_kwargs or {})
    }
    _ = fig_kwargs.pop('figsize', None)

    square_side = max(16, round(18 / 30 * proximity_matrix.shape[0]))
    bars_height = round(6 / 10 * len(scores))

    fig = plt.figure(figsize=(square_side * 2, square_side + bars_height), **fig_kwargs)
    gs = GridSpec(2, 4, height_ratios=[square_side, bars_height], figure=fig)
    axes = [
        fig.add_subplot(gs[0, :2]),
        fig.add_subplot(gs[0, 2:]),
        fig.add_subplot(gs[1, 1:3]),
    ]

    clustered_labels = list(flatten([c['columns'] for c in clusters]))

    comp.proximity(proximity_matrix, clusters=clusters, labels=labels, ax=axes[0])
    comp.cluster_scores(scores, clusters=clusters, labels=labels, ax=axes[1])
    comp.importance(scores, labels=[c['name'] for c in clusters], sort=True, ax=axes[2])
