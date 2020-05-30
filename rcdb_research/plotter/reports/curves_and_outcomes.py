import numpy as np

from typing import Optional, List

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .. import style
from ..utils import configure_axis

from .. import primitives as prim
from .. import components as comp


# Equities report
def curves_and_outcomes(curves: List[np.ndarray],
                        threshold: float = 0,
                        title: Optional[str] = 'Distributions of variables',
                        xlabel: Optional[str] = 'Datapoints',
                        ylabel: Optional[str] = 'Fraction',
                        fig_kwargs: Optional[dict] = None,
                        ax_kwargs: Optional[dict] = None,
                        pos_line_kwargs: Optional[dict] = None,
                        neg_line_kwargs: Optional[dict] = None,
                        pos_bar_kwargs: Optional[dict] = None,
                        neg_bar_kwargs: Optional[dict] = None):
    fig_kwargs = {**style.fig_kwargs(figsize=(16, 7), constrained_layout=True), **(fig_kwargs or {})}
    ax_kwargs = {
        **style.ax_kwargs(
            xformatter=ticker.FormatStrFormatter('%.0f'),
            yformatter=ticker.FormatStrFormatter('%.2f'),
        ),
        **(ax_kwargs or {})
    }
    pos_line_kwargs = {**style.line_kwargs(color='#49b4f2', alpha=0.1), **(pos_line_kwargs or {})}
    neg_line_kwargs = {**style.line_kwargs(color='#f27549', alpha=0.1), **(neg_line_kwargs or {})}
    pos_bar_kwargs = {**dict(color='#49b4f2', alpha=0.75), **(pos_bar_kwargs or {})}
    neg_bar_kwargs = {**dict(color='#f27549', alpha=0.75), **(neg_bar_kwargs or {})}

    outcomes = np.array([c[-1] for c in curves])
    pct_below_thr = outcomes[outcomes < threshold].size / outcomes.size

    # left, width = 0, 0.85
    # bottom, height = 0, 1
    # spacing = 0.01

    # rect_lines = [left, bottom, width, height]
    # rect_hist = [left + width + spacing, bottom, 1 - width - spacing, height]

    fig, (ax_lines, ax_hist) = plt.subplots(1, 2,
                                            gridspec_kw={'width_ratios': [5, 1], 'wspace': 0.05},
                                            **fig_kwargs)
    fig.set_constrained_layout_pads(w_pad=0., h_pad=0., hspace=0., wspace=0.001)
    # ax_lines = plt.axes(rect_lines)
    # ax_hist = plt.axes(rect_hist)

    configure_axis(ax_lines, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    cr_q975 = np.quantile(curves, 0.975, axis=0)
    cr_q750 = np.quantile(curves, 0.750, axis=0)
    cr_q500 = np.quantile(curves, 0.500, axis=0)
    cr_q250 = np.quantile(curves, 0.250, axis=0)
    cr_q025 = np.quantile(curves, 0.025, axis=0)

    comp.monte_carlo(curves,
                     threshold=threshold,
                     pos_line_kwargs=pos_line_kwargs,
                     neg_line_kwargs=neg_line_kwargs,
                     ax_kwargs=dict(ylocator=ticker.MaxNLocator(20)),
                     ax=ax_lines)

    prim.line(cr_q975, ax=ax_lines, line_kwargs=dict(color='#aaaaaa', linestyle='-', linewidth=3))
    prim.line(cr_q750, ax=ax_lines, line_kwargs=dict(color='#aaaaaa', linestyle='-', linewidth=3))
    prim.line(cr_q500, ax=ax_lines, line_kwargs=dict(color='#aaaaaa', linestyle='-', linewidth=3))
    prim.line(cr_q250, ax=ax_lines, line_kwargs=dict(color='#aaaaaa', linestyle='-', linewidth=3))
    prim.line(cr_q025, ax=ax_lines, line_kwargs=dict(color='#aaaaaa', linestyle='-', linewidth=3))

    yticks = ax_lines.get_yticks()
    ax_lines.set_xlim(-curves[0].size * 0.01, curves[0].size)
    ax_lines.set_ylim(yticks.min(), yticks.max())
    ax_hist.set_ylim(yticks.min(), yticks.max())

    prim.hist_pn(outcomes,
                 bins=yticks.size * 4,
                 threshold=threshold,
                 orientation='h',
                 thr_orientation='h',
                 pos_bar_kwargs=pos_bar_kwargs,
                 neg_bar_kwargs=neg_bar_kwargs,
                 ax_kwargs=dict(
                     tick_params=dict(bottom=False, left=False, labelbottom=False, labelleft=False),
                     ylocator=ticker.MaxNLocator(20)
                 ),
                 ax=ax_hist)

    ar_q975 = np.quantile(outcomes, 0.975, axis=0)
    ar_q750 = np.quantile(outcomes, 0.750, axis=0)
    ar_q500 = np.quantile(outcomes, 0.500, axis=0)
    ar_q250 = np.quantile(outcomes, 0.250, axis=0)
    ar_q025 = np.quantile(outcomes, 0.025, axis=0)

    ax_hist.axhline(ar_q975, color='#aaaaaa', lw=3, linestyle='-')
    ax_hist.axhline(ar_q750, color='#aaaaaa', lw=3, linestyle='-')
    ax_hist.axhline(ar_q500, color='#aaaaaa', lw=3, linestyle='-')
    ax_hist.axhline(ar_q250, color='#aaaaaa', lw=3, linestyle='-')
    ax_hist.axhline(ar_q025, color='#aaaaaa', lw=3, linestyle='-')

    ax_hist.axhline(y=threshold, linewidth=1, linestyle='--', color='black',
                    label=f'{pct_below_thr * 100:.2f}% below thr')

    legend_elements = [
        Line2D([0], [0], color='#aaaaaa', linestyle='-', lw=3, label=f'Q.975 = {ar_q975:.2f}'),
        Line2D([0], [0], color='#aaaaaa', linestyle='-', lw=3, label=f'Q.750 = {ar_q750:.2f}'),
        Line2D([0], [0], color='#aaaaaa', linestyle='-', lw=3, label=f'Q.500 = {ar_q500:.2f}'),
        Line2D([0], [0], color='#aaaaaa', linestyle='-', lw=3, label=f'Q.250 = {ar_q250:.2f}'),
        Line2D([0], [0], color='#aaaaaa', linestyle='-', lw=3, label=f'Q.025 = {ar_q025:.2f}'),
        Line2D([0], [0], color='#222222', linestyle='--', lw=2, label=f'below thr = {pct_below_thr * 100:.2f}%'),
    ]

    ax_lines.legend(handles=legend_elements, loc='best',
                    fancybox=False,
                    prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['labelsize']})
