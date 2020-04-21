import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from rcdb_research.plotter import style
from rcdb_research.plotter.utils import configure_axis


def distribution_colors(
        std_span='lightgray',
        std_border='gray',
        mean='gray',
        hist='#49b4f2',
        dots='#49b4f2',
        pdf='darkblue',
        kde='#f27549') -> dict: return locals()


def distribution_report(array: np.array, bins: int = 20, ticks=20,
                        title: Optional[str] = 'Distribution', xlabel: Optional[str] = 'Datapoints',
                        ylabel: Optional[str] = 'Density', colors: Optional[dict] = None,
                        fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
                        hist_kwargs: Optional[dict] = None, kde_kwargs: Optional[dict] = None,
                        pdf_kwargs: Optional[dict] = None,
                        ax=None):
    colors = colors or distribution_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 8))
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        tickrotation=45,
        xformatter=ticker.FormatStrFormatter('%.3f'),
        yformatter=ticker.FormatStrFormatter('%.0f'),
    )
    hist_kwargs = hist_kwargs or dict(alpha=0.9, zorder=100)
    kde_kwargs = kde_kwargs or dict(lw=3, alpha=0.85, zorder=102)
    pdf_kwargs = pdf_kwargs or dict(lw=3, alpha=0.85, zorder=102)
    _ = hist_kwargs.pop('color', None)
    _ = kde_kwargs.pop('color', None)
    _ = pdf_kwargs.pop('color', None)

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = (None, ax) if ax is not None else plt.subplots(**fig_kwargs)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    # Calculate mean and std
    mean = np.mean(array)
    std = np.std(array, ddof=1)

    # Plot histogram, KDE and PDF
    sns.distplot(array, bins=bins, norm_hist=True, kde=True, ax=axis, fit=scipy.stats.norm,
                 hist_kws={'color': colors['hist'], **hist_kwargs},
                 kde_kws={'color': colors['kde'], **kde_kwargs},
                 fit_kws={'color': colors['pdf'], **pdf_kwargs})

    axis.autoscale(axis='y')

    # Plot dots
    ylim = axis.get_ylim()
    offset_below_zero = 0.025
    dots_y = np.zeros_like(array) - offset_below_zero * (ylim[1] - ylim[0])
    axis.scatter(array, dots_y, color=colors['dots'], alpha=0.8)
    axis.set_ylim(-2 * offset_below_zero * (ylim[1] - ylim[0]), ylim[1])

    # Calculate zero level in axis coordinates
    ylim = axis.get_ylim()
    zerolvl = (0 - ylim[0]) / (ylim[1] - ylim[0])

    # Plot mean
    axis.axvline(mean, ymin=zerolvl, color=colors['mean'], linestyle='--', zorder=101)

    # Plot confidence intervals' spans
    std_alpha = 0.5
    axis.axvspan(mean - std, mean + std, ymin=zerolvl, color=colors['std_span'], alpha=std_alpha, zorder=3)
    axis.axvspan(mean - 2 * std, mean + 2 * std, ymin=zerolvl, color=colors['std_span'], alpha=std_alpha / 2, zorder=2)
    axis.axvspan(mean - 3 * std, mean + 3 * std, ymin=zerolvl, color=colors['std_span'], alpha=std_alpha / 3, zorder=1)

    # Plot confidence intervals' borders
    axis.axvline(mean - 1 * std, ymin=zerolvl, color=colors['std_border'], alpha=std_alpha, linestyle='-', zorder=101)
    axis.axvline(mean + 1 * std, ymin=zerolvl, color=colors['std_border'], alpha=std_alpha, linestyle='-', zorder=101)
    axis.axvline(mean - 2 * std, ymin=zerolvl, color=colors['std_border'], alpha=std_alpha / 2, linestyle='-', zorder=101)
    axis.axvline(mean + 2 * std, ymin=zerolvl, color=colors['std_border'], alpha=std_alpha / 2, linestyle='-', zorder=101)
    axis.axvline(mean - 3 * std, ymin=zerolvl, color=colors['std_border'], alpha=std_alpha / 3, linestyle='-', zorder=101)
    axis.axvline(mean + 3 * std, ymin=zerolvl, color=colors['std_border'], alpha=std_alpha / 3, linestyle='-', zorder=101)

    # Plot `ticks` number of xticks
    axis.xaxis.set_major_locator(ticker.MaxNLocator(ticks))

    # Setup and plot legend
    legend_elements = [
        Line2D([0], [0], color=colors['pdf'], lw=3, label=f'PDF, N[mean={mean:.3f}, std={std:.3f}]'),
        Line2D([0], [0], color=colors['kde'], lw=3, label='KDE, estimated'),
        Line2D([0], [0], color=colors['mean'], linestyle='--', lw=2, label=f'Mean, {mean:.3f}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['dots'], markersize=10,
               label=f'{xlabel}, {array.shape[0]}'),
        Patch(facecolor=colors['std_span'], alpha=std_alpha, edgecolor=colors['std_border'],
              label=f'68.0% confidence, [{mean - 1 * std:.3f}, {mean + 1 * std:.3f}]'),
        Patch(facecolor=colors['std_span'], alpha=std_alpha / 2, edgecolor=colors['std_border'],
              label=f'95.0% confidence, [{mean - 2 * std:.3f}, {mean + 2 * std:.3f}]'),
        Patch(facecolor=colors['std_span'], alpha=std_alpha / 3, edgecolor=colors['std_border'],
              label=f'99.7% confidence, [{mean - 3 * std:.3f}, {mean + 3 * std:.3f}]'),
    ]

    axis.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.45),
                fancybox=True, shadow=False, ncol=2,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['labelsize']})