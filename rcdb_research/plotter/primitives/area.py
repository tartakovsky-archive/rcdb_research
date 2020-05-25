import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def area(y1: np.array,
         y2: np.array,
         x: np.array = None,
         title: Optional[str] = None,
         xlabel: Optional[str] = None,
         ylabel: Optional[str] = None,
         line_kwargs: Optional[dict] = None,
         fig_kwargs: Optional[dict] = None,
         ax_kwargs: Optional[dict] = None,
         label: str = None,
         ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(), **(fig_kwargs or {})}
    ax_kwargs = {**style.ax_kwargs(), **(ax_kwargs or {})}
    line_kwargs = {**style.line_kwargs(color='#49b4f2'), **(line_kwargs or {})}
    _ = line_kwargs.pop('label', None)

    x = np.arange(y1.size) if x is None else x

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    axis.plot(x, y1, label=label, **line_kwargs)
    axis.plot(x, y2, **line_kwargs)
    axis.fill_between(x, y1, y2, facecolor=line_kwargs['color'], alpha=0.5)

    if ax is None:
        return fig, axis
