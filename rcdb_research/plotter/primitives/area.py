import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def area_colors(main: str = 'deepskyblue') -> dict: return locals()


def area(y1: np.array, y2: np.array, x: np.array = None,
         title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
         colors: Optional[dict] = None, line_kwargs: Optional[dict] = None,
         fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
         legend: str = None, ax=None):
    colors = colors or area_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs()
    line_kwargs = line_kwargs or style.line_kwargs()
    _ = line_kwargs.pop('color', None)
    _ = line_kwargs.pop('label', None)

    x = np.arange(y1.size) if x is None else x

    fig, axis = (None, ax) if ax is not None else plt.subplots(**fig_kwargs)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    axis.plot(x, y1, color=colors['main'], label=legend, **line_kwargs)
    axis.plot(x, y2, color=colors['main'], **line_kwargs)
    axis.fill_between(x, y1, y2, facecolor=colors['main'], alpha=0.5)

    if ax is None:
        return fig, axis
