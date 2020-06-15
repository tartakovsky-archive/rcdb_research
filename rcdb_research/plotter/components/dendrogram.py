import numpy as np

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import ticker

from .. import style
from ..utils import configure_axis

from scipy.cluster.hierarchy import dendrogram as plot_dendrogram


def dendrogram(model,
               names: Optional[list] = None,
               orientation='right',
               title: Optional[str] = 'Dendrogram',
               xlabel: Optional[str] = 'Distance',
               ylabel: Optional[str] = 'Feature',
               fig_kwargs: Optional[dict] = None,
               ax_kwargs: Optional[dict] = None,
               dendrogram_kwargs: Optional[dict] = None,
               ax=None):
    h_or_v = 'h' if orientation == 'right' or orientation == 'left' else 'v'

    fig_kwargs = {
        **style.fig_kwargs(
            figsize=(16, 10) if h_or_v == 'h' else (16, 14)
        ),
        **(fig_kwargs or {})
    }
    ax_kwargs = {
        **style.ax_kwargs(
            yformatter=None,
            tickrotation=90 if h_or_v == 'v' else 0,
            xlocator=ticker.MaxNLocator(21) if h_or_v == 'h' else None,
            ylocator=ticker.MaxNLocator(21) if h_or_v == 'v' else None,
        ),
        **(ax_kwargs or {})
    }
    dendrogram_kwargs = {**(dendrogram_kwargs or {})}
    _ = dendrogram_kwargs
    # Use external axis or create one
    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # If we don't have this information, use a uniform one for plotting
    if hasattr(model, 'distances_'):
        distance = model.distances_
    else:
        distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0] + 2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    plot_dendrogram(linkage_matrix, labels=names, color_threshold=model.distance_threshold, orientation=orientation, ax=axis, **dendrogram_kwargs)

    if h_or_v == 'h':
        if model.distance_threshold is not None:
            axis.axvline(model.distance_threshold, color='black', linestyle='--')
        configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)
        axis.set_yticklabels(names)
    else:
        if model.distance_threshold is not None:
            axis.axhline(model.distance_threshold, color='black', linestyle='--')
        configure_axis(axis, title, ylabel, xlabel, ax_kwargs=ax_kwargs)
        axis.set_xticklabels(names)

    if ax is None:
        return fig, axis
