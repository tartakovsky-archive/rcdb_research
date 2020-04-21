import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def bars_colors(pos: str = '#49b4f2', neg: str = '#f27549') -> dict: return locals()


def bars_legend(pos: Optional[str] = None, neg: Optional[str] = None) -> dict: return locals()


def bars(y: np.array, x: np.array = None, threshold: float = 0,
         title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
         colors: Optional[dict] = None, legend: Optional[dict] = None,
         fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
         bar_kwargs: Optional[dict] = None,
         ax=None):
    colors = colors or bars_colors()
    legend = legend or bars_legend()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs()
    bar_kwargs = bar_kwargs or style.line_kwargs()
    _ = bar_kwargs.pop('color', None)
    _ = bar_kwargs.pop('label', None)

    x = np.arange(y.size) if x is None else x

    fig, axis = (None, ax) if ax is not None else plt.subplots(**fig_kwargs)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    if np.any(y >= threshold):
        y_pos = np.where(y >= threshold, y - threshold, np.nan)
        axis.bar(x, height=y_pos, bottom=threshold,
                 color=colors['pos'], label=legend['pos'], **bar_kwargs)

    if np.any(y < threshold):
        y_neg = np.where(y < threshold, y - threshold, np.nan)
        axis.bar(x, height=y_neg, bottom=threshold,
                 color=colors['neg'], label=legend['neg'], **bar_kwargs)

    if ax is None:
        return fig, axis
