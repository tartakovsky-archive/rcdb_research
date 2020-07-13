import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis, second_index, datestring


def line(y,
         x=None,
         title: Optional[str] = None,
         xlabel: Optional[str] = None,
         ylabel: Optional[str] = None,
         fig_kwargs: Optional[dict] = None,
         ax_kwargs: Optional[dict] = None,
         line_kwargs: Optional[dict] = None,
         fill: bool = False,
         show_dates: bool = False,
         ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(), **(fig_kwargs or {})}
    ax_kwargs = {**style.ax_kwargs(), **(ax_kwargs or {})}
    line_kwargs = {**style.line_kwargs(color='#49b4f2'), **(line_kwargs or {})}

    x = np.arange(y.size) if x is None else x

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    axis.plot(x, y, **line_kwargs)

    if fill:
        axis.fill_between(x, y, facecolor=line_kwargs['color'], alpha=0.65)

    if show_dates:
        axis.set_xlabel(None)
        second_index(axis, datestring(y.index), xlabel=xlabel, ax_kwargs=ax_kwargs)

    if ax is None:
        return fig, axis
