import numpy as np

from typing import Optional, List

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch

from .. import style
from ..utils import configure_axis

from .. import primitives as prim
from .. import components as comp


# Equities report
def equities(curves: List[np.ndarray],
             threshold: float = 0,
             title: Optional[str] = 'Distributions of variables',
             xlabel: Optional[str] = 'Datapoints',
             ylabel: Optional[str] = 'Fraction',
             fig_kwargs: Optional[dict] = None,
             ax_kwargs: Optional[dict] = None):
    fig_kwargs = {**style.fig_kwargs(figsize=(16, 7), constrained_layout=True), **(fig_kwargs or {})}
    ax_kwargs = {
        **style.ax_kwargs(
            xformatter=ticker.FormatStrFormatter('%.0f'),
            yformatter=ticker.FormatStrFormatter('%.2f'),
        ),
        **(ax_kwargs or {})
    }

    abs_rets = np.array([c[-1] for c in curves])
    pct_below_thr = abs_rets[abs_rets < threshold].size / abs_rets.size

    left, width = 0, 0.85
    bottom, height = 0, 1
    spacing = 0.01

    rect_lines = [left, bottom, width, height]
    rect_hist = [left + width + spacing, bottom, 1 - width - spacing, height]

    fig = plt.figure(**fig_kwargs)
    ax_lines = plt.axes(rect_lines)
    ax_hist = plt.axes(rect_hist)

    configure_axis(ax_lines, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    comp.monte_carlo(curves,
                     threshold=0.0,
                     plot_mean=False,
                     plot_median=True,
                     pos_line_kwargs={'alpha': 0.1, 'color': 'blue'},
                     neg_line_kwargs={'alpha': 0.1, 'color': 'red'},
                     median_kwargs=dict(color='silver'),
                     ax_kwargs=dict(ylocator=ticker.MaxNLocator(20)),
                     ax=ax_lines)

    yticks = ax_lines.get_yticks()
    ax_lines.set_xlim(-curves[0].size * 0.01, curves[0].size)
    ax_lines.set_ylim(yticks.min(), yticks.max())
    ax_hist.set_ylim(yticks.min(), yticks.max())

    hist, bins = np.histogram(abs_rets, bins=yticks.size * 4)
    width = 1.0 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    prim.bars_pn(x=center,
                 y=hist,
                 threshold=0.0,
                 width=width,
                 orientation='h',
                 thr_orientation='h',
                 pos_bar_kwargs=dict(color='blue', alpha=0.75),
                 neg_bar_kwargs=dict(color='red', alpha=0.75),
                 ax_kwargs=dict(
                     tick_params=dict(bottom=False, left=False, labelbottom=False, labelleft=False),
                     ylocator=ticker.MaxNLocator(20)
                 ),
                 ax=ax_hist)

    median = np.median(abs_rets)
    q25 = np.quantile(abs_rets, 0.25)
    q75 = np.quantile(abs_rets, 0.75)
    q025 = np.quantile(abs_rets, 0.025)
    q975 = np.quantile(abs_rets, 0.975)

    # ax_hist.axhline(np.mean(ar), color='bisque', lw=4, linestyle='--', label='mean')
    ax_hist.axhline(median, color='#aaaaaa', lw=3, linestyle='-', label=f'median = {median:.2f}')
    ax_hist.axhline(q25, color='#aaaaaa', lw=3, linestyle='--', label=f'Q.25   = {q25:.2f}')
    ax_hist.axhline(q75, color='#aaaaaa', lw=3, linestyle='--', label=f'Q.75   = {q75:.2f}')
    ax_hist.axhline(q025, color='#aaaaaa', lw=3, linestyle='-.', label=f'Q.025  = {q025:.2f}')
    ax_hist.axhline(q975, color='#aaaaaa', lw=3, linestyle='-.', label=f'Q.975  = {q975:.2f}')
    ax_hist.legend(loc='upper right',
                   fancybox=False,
                   prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['labelsize']})
