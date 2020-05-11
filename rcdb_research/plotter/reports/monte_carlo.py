import matplotlib.pyplot as plt
# from matplotlib import ticker
from typing import List, Optional
import numpy as np

# from .. import primitives
from ..utils import configure_axis

from .. import style
# from .. import utils


#
# def plot_equities(equities, plot_avg=False, avg_only=False, ylim=(0, 1)):
#     size = len(equities[0])
#     fig, ax = plt.subplots(1, figsize=(15, 6))
#     if not avg_only:
#         for e in equities:
#             ax.plot(e, linewidth=2)
#     if plot_avg or avg_only:
#         mean = np.array(equities).mean(axis=0)
#         ax.plot(mean, linewidth=4, color='red')
#     ax.axhline(linewidth=1, linestyle='--', color='black')
# #     ax.set_ylim(ylim[0], ylim[1])
#     plt.tight_layout()
#     plt.show()
#

def monte_carlo_report(curves: List[np.ndarray], mean_curve=False, mean_only=False,
                       title: Optional[str] = None,
                       xlabel: Optional[str] = 'Observations',
                       ylabel: Optional[str] = 'Cumulative return',
                       fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
                       line_kwargs: Optional[dict] = None):
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 7))
    ax_kwargs = ax_kwargs or style.ax_kwargs()
    line_kwargs = line_kwargs or style.line_kwargs(linewidth=2)
    _ = line_kwargs.pop('color', None)

    fig, axis = plt.subplots(**fig_kwargs)
    fig.suptitle("Monte Carlo report", x=0.526, y=1.05, **style.suptitle_kwargs())
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    # plot lines
    if not mean_only:
        for curve in curves:
            axis.plot(curve, **line_kwargs)

    if mean_curve or mean_only:
        mean = np.array(curves).mean(axis=0)
        axis.plot(mean, linewidth=4, color='red')

    axis.axhline(linewidth=1, linestyle='--', color='black')
