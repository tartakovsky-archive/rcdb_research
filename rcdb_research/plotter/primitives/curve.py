import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def curve_colors(pos: str = '#49b4f2', neg: str = '#f27549') -> dict:
    return locals()


def curve_legend(pos: Optional[str] = None, neg: Optional[str] = None) -> dict:
    return locals()


def curve(y: np.array, x: np.array = None, threshold: float = 0, title: Optional[str] = None,
          xlabel: Optional[str] = None, ylabel: Optional[str] = None,
          colors: Optional[dict] = None, legend: Optional[dict] = None,
          fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
          line_kwargs: Optional[dict] = None,
          fill: bool = False, ax=None) -> Optional[tuple]:
    # Validate arguments
    colors = colors or curve_colors()
    legend = legend or curve_legend()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs()
    line_kwargs = line_kwargs or style.line_kwargs()
    _ = line_kwargs.pop('color', None)
    _ = line_kwargs.pop('label', None)

    x = np.arange(y.size) if x is None else x

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    if np.any(y >= threshold):
        y_pos = np.where(y >= threshold, y, np.nan)
        y_pos[_edges_of_nans(y_pos)] = threshold
        axis.plot(x, y_pos, color=colors['pos'], label=legend['pos'], **line_kwargs)
        if fill:
            axis.fill_between(x, threshold, y_pos, facecolor=colors['pos'], alpha=0.65)

    if np.any(y < threshold):
        y_neg = np.where(y < threshold, y, np.nan)
        y_neg[_edges_of_nans(y_neg)] = threshold
        axis.plot(x, y_neg, color=colors['neg'], label=legend['neg'], **line_kwargs)
        if fill:
            axis.fill_between(x, threshold, y_neg, facecolor=colors['neg'], alpha=0.65)

    if ax is None:
        return fig, axis


def _edges_of_nans(array: np.array):
    # display(array)
    # > [1, nan, nan, 2, 3, nan, 1, nan, nan, nan]
    isnan = np.concatenate(([0], np.isnan(array), [0]))
    # > [0 0 1 1 0 0 1 0 1 1 1 0]
    changes = np.abs(np.diff(isnan))
    # > [0 1 0 1 0 1 1 1 0 0 1]
    ranges = np.where(changes == 1)[0].reshape(-1, 2)
    # > [[ 1  3], [ 5  6], [ 7 10]]
    ranges[:, 1] = ranges[:, 1] - 1
    # > [[1 2], [5 5], [7 9]]
    edges = np.unique(ranges.ravel())
    # > [1 2 5 7 9]
    return edges
