import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import List, Optional
import numpy as np

from ..utils import configure_axis, second_index, datestring
from .. import style
from ..primitives import line_pn


def monte_carlo(curves: list,
                x=None,
                plot_mean=False,
                plot_median=False,
                threshold=None,
                title: Optional[str] = None,
                xlabel: Optional[str] = 'Observations',
                ylabel: Optional[str] = 'Cumulative return',
                fig_kwargs: Optional[dict] = None,
                ax_kwargs: Optional[dict] = None,
                line_kwargs: Optional[dict] = None,
                pos_line_kwargs: Optional[dict] = None,
                neg_line_kwargs: Optional[dict] = None,
                mean_kwargs: Optional[dict] = None,
                median_kwargs: Optional[dict] = None,
                show_dates: bool = False,
                ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(figsize=(16, 7)), **(fig_kwargs or {})}
    ax_kwargs = {
        **style.ax_kwargs(
            xformatter=ticker.FormatStrFormatter('%.0f'),
        ),
        **(ax_kwargs or {})
    }

    line_kwargs = {**style.line_kwargs(), **(line_kwargs or {})}
    mean_kwargs = {**style.line_kwargs(linewidth=4, color='red'), **(mean_kwargs or {})}
    median_kwargs = {**style.line_kwargs(linewidth=4, color='orange'), **(median_kwargs or {})}

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    x = np.arange(curves[0].size) if x is None else x

    # plot lines
    for c in curves:
        if threshold is None:
            axis.plot(x, c, **line_kwargs)
        else:
            line_pn(c, threshold=threshold, pos_line_kwargs=pos_line_kwargs,
                    neg_line_kwargs=neg_line_kwargs, ax=axis)

    if plot_mean:
        mean = np.mean(curves, axis=0)
        axis.plot(mean, **mean_kwargs)

    if plot_median:
        median = np.median(curves, axis=0)
        axis.plot(median, **median_kwargs)

    axis.axhline(y=threshold or 0, linewidth=1, linestyle='--', color='black')

    if show_dates:
        axis.set_xlabel(None)
        second_index(axis, datestring(curves[0].index), xlabel=xlabel, ax_kwargs=ax_kwargs)

    if ax is None:
        return fig, axis
