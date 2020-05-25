import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def bars_pn(y: np.array,
            x: np.array = None,
            threshold: float = 0,
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            fig_kwargs: Optional[dict] = None,
            ax_kwargs: Optional[dict] = None,
            pos_bar_kwargs: Optional[dict] = None,
            neg_bar_kwargs: Optional[dict] = None,
            ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(), **(fig_kwargs or {})}
    ax_kwargs = {**style.ax_kwargs(), **(ax_kwargs or {})}
    pos_bar_kwargs = {**style.line_kwargs(color='#49b4f2'), **(pos_bar_kwargs or {})}
    neg_bar_kwargs = {**style.line_kwargs(color='#f27549'), **(neg_bar_kwargs or {})}

    x = np.arange(y.size) if x is None else x

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    if np.any(y >= threshold):
        y_pos = np.where(y >= threshold, y - threshold, np.nan)
        axis.bar(x, height=y_pos, bottom=threshold, **pos_bar_kwargs)

    if np.any(y < threshold):
        y_neg = np.where(y < threshold, y - threshold, np.nan)
        axis.bar(x, height=y_neg, bottom=threshold, **neg_bar_kwargs)

    if ax is None:
        return fig, axis
