import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def line_pn(y: np.array,
            x: np.array = None,
            threshold: float = 0,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            fig_kwargs: Optional[dict] = None,
            ax_kwargs: Optional[dict] = None,
            pos_line_kwargs: Optional[dict] = None,
            neg_line_kwargs: Optional[dict] = None,
            fill: bool = False, ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(), **(fig_kwargs or {})}
    ax_kwargs = {**style.ax_kwargs(), **(ax_kwargs or {})}
    pos_line_kwargs = {**style.line_kwargs(color='#49b4f2'), **(pos_line_kwargs or {})}
    neg_line_kwargs = {**style.line_kwargs(color='#f27549'), **(neg_line_kwargs or {})}

    x = np.arange(y.size) if x is None else x

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    if np.any(y >= threshold):
        y_pos = np.where(y >= threshold, y, np.nan)
        y_pos[_edges_of_nans(y_pos)] = threshold
        axis.plot(x, y_pos, **pos_line_kwargs)
        if fill:
            axis.fill_between(x, threshold, y_pos, facecolor=pos_line_kwargs['color'], alpha=0.65)

    if np.any(y < threshold):
        y_neg = np.where(y < threshold, y, np.nan)
        y_neg[_edges_of_nans(y_neg)] = threshold
        axis.plot(x, y_neg, **neg_line_kwargs)
        if fill:
            axis.fill_between(x, threshold, y_neg, facecolor=neg_line_kwargs['color'], alpha=0.65)

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
