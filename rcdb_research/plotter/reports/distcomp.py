import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from rcdb_research.plotter import style
from rcdb_research.plotter.utils import configure_axis


def distcomp_colors(
        std1_span='dodgerblue',
        std1_border='dodgerblue',
        mean1='dodgerblue',
        hist1='dodgerblue',
        dots1='dodgerblue',
        std2_span='lightsalmon',
        std2_border='lightsalmon',
        mean2='lightsalmon',
        hist2='lightsalmon',
        dots2='lightsalmon') -> dict: return locals()


# Distribution Comparison Report
def distcomp_report(a: np.ndarray, b: np.ndarray,
                    bins: int = 20, ticks=20,
                    title: Optional[str] = 'Distributions of variables A and B',
                    xlabel: Optional[str] = '(A) Datapoints / (B) Datapoints',
                    ylabel: Optional[str] = 'Density', colors: Optional[dict] = None,
                    fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
                    hist_kwargs: Optional[dict] = None, kde_kwargs: Optional[dict] = None,
                    ax=None):
    colors = colors or distcomp_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs(tickrotation=45)
    hist_kwargs = hist_kwargs or dict(alpha=0.7, zorder=100)
    kde_kwargs = kde_kwargs or dict(lw=3, alpha=0.85, zorder=102)
    _ = hist_kwargs.pop('color', None)
    _ = kde_kwargs.pop('color', None)

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = (None, ax) if ax is not None else plt.subplots(**fig_kwargs)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs,
                   xformatter=ticker.FormatStrFormatter('%.3f'),
                   yformatter=ticker.FormatStrFormatter('%.0f'))

    # Plot histograms and KDEs
    sns.distplot(a, bins=bins, norm_hist=True, kde=True, ax=axis,
                 hist_kws={'color': colors['hist1'], **hist_kwargs},
                 kde_kws={'color': colors['hist1'], **kde_kwargs})
    sns.distplot(b, bins=bins, norm_hist=True, kde=True, ax=axis,
                 hist_kws={'color': colors['hist2'], **hist_kwargs},
                 kde_kws={'color': colors['hist2'], **kde_kwargs})

    # Calculate means and stds
    mean1 = np.mean(a)
    mean2 = np.mean(b)
    std1 = np.std(a, ddof=1)
    std2 = np.std(b, ddof=1)

    # Plot dots
    ylim = ax.get_ylim()
    offset_below_zero = 0.025
    dots1_y = np.zeros_like(a) - offset_below_zero * (ylim[1] - ylim[0])
    dots2_y = np.zeros_like(b) - 2 * offset_below_zero * (ylim[1] - ylim[0])
    ax.scatter(a, dots1_y, color=colors['dots1'], alpha=0.8)
    ax.scatter(b, dots2_y, color=colors['dots2'], alpha=0.8)
    ax.set_ylim(-3 * offset_below_zero * (ylim[1] - ylim[0]), ylim[1])

    # Calculate zero level in axis coordinates
    ylim = ax.get_ylim()
    zerolvl = (0 - ylim[0]) / (ylim[1] - ylim[0])

    # Plot means
    ax.axvline(mean1, ymin=zerolvl, color=colors['mean1'], lw=3, linestyle='--', zorder=101)
    ax.axvline(mean2, ymin=zerolvl, color=colors['mean2'], lw=3, linestyle='--', zorder=101)

    # Plot confidence intervals' spans
    std_alpha = 0.2
    ax.axvspan(mean1 - 2 * std1, mean1 + 2 * std1, ymin=zerolvl, color=colors['std1_span'], alpha=std_alpha, zorder=1)
    ax.axvspan(mean2 - 2 * std2, mean2 + 2 * std2, ymin=zerolvl, color=colors['std2_span'], alpha=std_alpha, zorder=1)

    # Plot confidence intervals' borders
    ax.axvline(mean1 - 2 * std1, ymin=zerolvl, color=colors['std1_border'], alpha=1, linestyle='-', zorder=101)
    ax.axvline(mean1 + 2 * std1, ymin=zerolvl, color=colors['std1_border'], alpha=1, linestyle='-', zorder=101)
    ax.axvline(mean2 - 2 * std2, ymin=zerolvl, color=colors['std2_border'], alpha=1, linestyle='-', zorder=101)
    ax.axvline(mean2 + 2 * std2, ymin=zerolvl, color=colors['std2_border'], alpha=1, linestyle='-', zorder=101)

    # Plot `ticks` number of xticks
    axis.xaxis.set_major_locator(ticker.MaxNLocator(ticks))

    # Setup and plot legend
    legend_elements = [
        Line2D([0], [0], color=colors['hist1'], lw=3, label='(A) KDE, estimated'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['dots1'], markersize=10,
               label=f'(A) Datapoints, {a.shape[0]}'),
        Line2D([0], [0], color=colors['mean1'], linestyle='--', lw=3, label=f'(A) Mean, {mean1:.3f}'),
        Patch(facecolor=colors['std1_span'], alpha=std_alpha, edgecolor=colors['std1_border'],
              label=f'(A) 95.0% confidence, [{mean1 - 2 * std1:.3f}, {mean1 + 2 * std1:.3f}]'),

        Line2D([0], [0], color=colors['hist2'], lw=3, label='(B) KDE, estimated'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['dots2'], markersize=10,
               label=f'(B) Datapoints, {a.shape[0]}'),
        Line2D([0], [0], color=colors['mean2'], linestyle='--', lw=3, label=f'(B) Mean, {mean2:.3f}'),
        Patch(facecolor=colors['std2_span'], alpha=std_alpha, edgecolor=colors['std2_border'],
              label=f'(B) 95.0% confidence, [{mean2 - 2 * std2:.3f}, {mean2 + 2 * std2:.3f}]'),
    ]

    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.4),
              fancybox=True, shadow=False, ncol=2,
              prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['labelsize']})
