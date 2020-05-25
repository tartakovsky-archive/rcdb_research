import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Optional

from .. import style
from ..utils import configure_axis


def hist_colors(main: str = '#49b4f2') -> dict:
    return locals()


def hist(array: np.array,
         bins: int = 20,
         ticks=20,
         title: Optional[str] = 'Histogram',
         xlabel: Optional[str] = 'Value bins',
         ylabel: Optional[str] = 'Count',
         fig_kwargs: Optional[dict] = None,
         ax_kwargs: Optional[dict] = None,
         hist_kwargs: Optional[dict] = None,
         ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(), **(fig_kwargs or {})}
    ax_kwargs = {
        **style.ax_kwargs(
            tickrotation=45,
            xlocator=ticker.MaxNLocator(ticks)
        ),
        **(ax_kwargs or {})
    }
    hist_kwargs = {**style.hist_kwargs(color='#49b4f2'), **(hist_kwargs or {})}

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    axis.hist(array, bins=bins, **hist_kwargs)

    if ax is None:
        return fig, axis
