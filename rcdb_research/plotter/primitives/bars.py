import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def bars(y: np.array,
         x: np.array = None,
         title: Optional[str] = None,
         xlabel: Optional[str] = None,
         ylabel: Optional[str] = None,
         fig_kwargs: Optional[dict] = None,
         ax_kwargs: Optional[dict] = None,
         bar_kwargs: Optional[dict] = None,
         ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(), **(fig_kwargs or {})}
    ax_kwargs = {**style.ax_kwargs(), **(ax_kwargs or {})}
    bar_kwargs = {**dict(color='#49b4f2'), **(bar_kwargs or {})}

    x = np.arange(y.size) if x is None else x

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    axis.bar(x, height=y, **bar_kwargs)

    if ax is None:
        return fig, axis
