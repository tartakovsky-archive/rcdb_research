import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Optional

from .. import style
from ..utils import configure_axis


def hist_colors(main: str = '#49b4f2') -> dict:
    return locals()


def hist(array: np.array, bins: int = 20, ticks=20,
         title: Optional[str] = 'Histogram', xlabel: Optional[str] = 'Value bins',
         ylabel: Optional[str] = 'Count', colors: Optional[dict] = None,
         fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
         hist_kwargs: Optional[dict] = None,
         ax=None):
    colors = colors or hist_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs(tickrotation=45)
    _ = ax_kwargs.pop('xlocator')
    hist_kwargs = hist_kwargs or style.hist_kwargs()
    _ = hist_kwargs.pop('color', None)

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    axis.xaxis.set_major_locator(ticker.MaxNLocator(ticks))

    axis.hist(array, bins=bins, color=colors['main'], **hist_kwargs)

    if ax is None:
        return fig, axis
