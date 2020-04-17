import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def line_colors(main: str = 'deepskyblue') -> dict: return locals()


def line(y: np.array, x: np.array = None, title: Optional[str] = None,
         xlabel: Optional[str] = None, ylabel: Optional[str] = None,
         colors: Optional[dict] = None, legend: Optional[str] = None,
         fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
         line_kwargs: Optional[dict] = None,
         fill: bool = False, ax=None) -> Optional[tuple]:
    # Validate arguments
    colors = colors or line_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs()
    line_kwargs = line_kwargs or style.line_kwargs()
    _ = line_kwargs.pop('color', None)
    _ = line_kwargs.pop('label', None)

    x = np.arange(y.size) if x is None else x

    fig, axis = (None, ax) if ax is not None else plt.subplots(**fig_kwargs)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    axis.plot(x, y, color=colors['main'], label=legend, **line_kwargs)
    if fill:
        axis.fill_between(x, y, facecolor=colors['main'], alpha=0.5)

    if ax is None:
        return fig, axis
