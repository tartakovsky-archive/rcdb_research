import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import List, Optional
import numpy as np

from rcdb_research.plotter.utils import configure_axis

from rcdb_research.plotter import style


def monte_carlo(curves: List[np.ndarray],
                mean_curve=False,
                mean_only=False,
                title: Optional[str] = None,
                xlabel: Optional[str] = 'Observations',
                ylabel: Optional[str] = 'Cumulative return',
                fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
                line_kwargs: Optional[dict] = None,
                ax=None) -> Optional[tuple]:
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 7))
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        xformatter=ticker.FormatStrFormatter('%.0f'),
    )
    line_kwargs = line_kwargs or style.line_kwargs(linewidth=2)
    _ = line_kwargs.pop('color', None)

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    # plot lines
    if not mean_only:
        for curve in curves:
            axis.plot(curve, **line_kwargs)

    if mean_curve or mean_only:
        mean = np.array(curves).mean(axis=0)
        axis.plot(mean, linewidth=4, color='red')

    axis.axhline(linewidth=1, linestyle='--', color='black')

    if ax is None:
        return fig, axis
