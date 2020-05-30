import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .. import style
from ..utils import configure_axis


def bars_pn(y: np.array,
            x: np.array = None,
            width: float = 1.0,
            threshold: float = 0,
            orientation: str = 'v',
            thr_orientation: str = 'v',
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            fig_kwargs: Optional[dict] = None,
            ax_kwargs: Optional[dict] = None,
            pos_bar_kwargs: Optional[dict] = None,
            neg_bar_kwargs: Optional[dict] = None,
            ax=None) -> Optional[tuple]:
    supported_orientations = ['v', 'h', 'vertical', 'horizontal']
    if orientation not in supported_orientations:
        raise ValueError(
            f'{orientation} orientation is not supported. Should be one of the following: {supported_orientations}'
        )
    if thr_orientation not in supported_orientations:
        raise ValueError(
            f'{thr_orientation} thr_orientation is not supported. '
            f'Should be one of the following: {supported_orientations}'
        )
    if orientation == 'v':
        orientation = 'vertical'
    elif orientation == 'h':
        orientation = 'horizontal'

    if thr_orientation == 'v':
        thr_orientation = 'vertical'
    elif thr_orientation == 'h':
        thr_orientation = 'horizontal'

    fig_kwargs = {**style.fig_kwargs(), **(fig_kwargs or {})}
    ax_kwargs = {**style.ax_kwargs(), **(ax_kwargs or {})}
    pos_bar_kwargs = {**dict(color='#49b4f2'), **(pos_bar_kwargs or {})}
    neg_bar_kwargs = {**dict(color='#f27549'), **(neg_bar_kwargs or {})}

    x = np.arange(y.size) if x is None else x

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    # Based on `orientation` and `thr_orientation` parameters there are four possible plots

    if orientation == 'vertical' and thr_orientation == 'vertical':
        if np.any(x >= threshold):
            axis.bar(x=x[x >= threshold], height=y[x >= threshold], width=width, **pos_bar_kwargs)
        if np.any(x < threshold):
            axis.bar(x=x[x < threshold], height=y[x < threshold], width=width, **neg_bar_kwargs)

    elif orientation == 'vertical' and thr_orientation == 'horizontal':
        if np.any(y >= threshold):
            axis.bar(x=x[y >= threshold], height=y[y >= threshold] - threshold,
                     bottom=threshold, width=width, **pos_bar_kwargs)
        if np.any(y < threshold):
            axis.bar(x=x[y < threshold], height=y[y < threshold] - threshold,
                     bottom=threshold, width=width, **neg_bar_kwargs)

    elif orientation == 'horizontal' and thr_orientation == 'horizontal':
        if np.any(x >= threshold):
            axis.barh(y=x[x >= threshold], width=y[x >= threshold], height=width, **pos_bar_kwargs)
        if np.any(x < threshold):
            axis.barh(y=x[x < threshold], width=y[x < threshold], height=width, **neg_bar_kwargs)

    elif orientation == 'horizontal' and thr_orientation == 'vertical':
        if np.any(y >= threshold):
            axis.barh(y=x[y >= threshold], width=y[y >= threshold] - threshold,
                      left=threshold, height=width, **pos_bar_kwargs)
        if np.any(y < threshold):
            axis.barh(y=x[y < threshold], width=y[y < threshold] - threshold,
                      left=threshold, height=width, **neg_bar_kwargs)

    if thr_orientation == 'vertical':
        axis.axvline(x=threshold, linewidth=1, linestyle='--', color='black')
    elif thr_orientation == 'horizontal':
        axis.axhline(y=threshold, linewidth=1, linestyle='--', color='black')

    if ax is None:
        return fig, axis
