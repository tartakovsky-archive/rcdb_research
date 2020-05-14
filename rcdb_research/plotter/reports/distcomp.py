import numpy as np
import seaborn as sns

from typing import Optional, List

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .. import style
from ..utils import configure_axis


def distcomp_colors(
        span='lightgray',
        a='#49b4f2',
        b='#f27549',
        baseline='#666666',
) -> dict:
    return locals()


# Distribution Comparison Report
def distcomp_report(a: np.ndarray,
                    b: Optional[np.ndarray] = None,
                    baseline: Optional[np.ndarray] = None,
                    a_name: str = '1',
                    b_name: str = '2',
                    baseline_name: str = 'Baseline',
                    confint_n_std: float = 2.0,
                    bins: int = 20,
                    ticks: int = 20,
                    title: Optional[str] = 'Distributions of variables',
                    xlabel: Optional[str] = 'Datapoints',
                    ylabel: Optional[str] = 'Frequency',
                    colors: Optional[dict] = None,
                    fig_kwargs: Optional[dict] = None,
                    ax_kwargs: Optional[dict] = None,
                    hist_kwargs: Optional[dict] = None,
                    kde_kwargs: Optional[dict] = None,
                    ax=None):
    colors = colors or distcomp_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 7))
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        tickrotation=45,
        xformatter=ticker.FormatStrFormatter('%.3f'),
        yformatter=ticker.FormatStrFormatter('%.2f'),
    )

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = (None, ax) if ax is not None else plt.subplots(**fig_kwargs)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    legend_elements = []
    arrays = [a]
    names = [a_name]
    cs = [colors['a']]
    if b is not None:
        arrays.append(b)
        names.append(b_name)
        cs.append(colors['b'])
    if baseline is not None:
        arrays.append(baseline)
        names.append(baseline_name)
        cs.append(colors['baseline'])

    # plot histograms
    for i, array in enumerate(arrays):
        if array is not baseline:
            j = i + 1
            hist_kwargs = hist_kwargs or dict(alpha=0.85, zorder=j * 100)
            kde_kwargs = kde_kwargs or dict(lw=3, alpha=0.7, zorder=j * 102)
            _ = hist_kwargs.pop('color', None)
            _ = kde_kwargs.pop('color', None)

            sns.distplot(array, bins=bins, norm_hist=False, kde=True, ax=axis,
                         hist_kws={'color': cs[i], **hist_kwargs},
                         kde_kws={'color': cs[i], **kde_kwargs})

    # plot spans

    for i, array in enumerate(arrays):
        j = i + 1

        # Calculate means and stds
        mean = np.mean(array)
        std = np.std(array, ddof=1)

        # Plot dot or box plots
        if array is not baseline:
            ylim = axis.get_ylim()
            offset_below_zero = 0.03
            offset_mult = len(arrays) if baseline is not None else len(arrays) + 1

            dots_y = np.zeros_like(array) - j * offset_below_zero * (ylim[1] - ylim[0])

            if a.size < 100 and b.size < 100:
                axis.scatter(array, dots_y, color=cs[i])
            else:
                axis.boxplot(
                    array,
                    positions=[dots_y[0]],
                    patch_artist=True,
                    notch=False,
                    widths=(ylim[1] - ylim[0]) * 0.017,
                    vert=False,
                    manage_ticks=False,
                    showmeans=True,
                    meanline=True,
                    medianprops=dict(lw=0),
                    meanprops=dict(color='#666666', lw=1, linestyle='-', alpha=0.5),
                    boxprops=dict(facecolor=cs[i], color=cs[i]),
                    whiskerprops=dict(color=cs[i], lw=2),
                    capprops=dict(color=cs[i], lw=2),
                    flierprops=dict(markersize=5, markeredgecolor=cs[i], alpha=0.8),
                )

            axis.set_ylim(-offset_mult * offset_below_zero * (ylim[1] - ylim[0]), ylim[1])

        # Calculate zero level in axis coordinates
        ylim = axis.get_ylim()
        zerolvl = (0 - ylim[0]) / (ylim[1] - ylim[0])

        # Plot means
        axis.axvline(mean, ymin=zerolvl, color=cs[i], lw=3, linestyle='--', zorder=j * 110)

        # Plot confidence intervals' spans
        std_alpha = 0.2
        axis.axvspan(mean - confint_n_std * std, mean + confint_n_std * std, ymin=zerolvl, color=colors['span'], alpha=std_alpha, zorder=1)

        # Plot confidence intervals' borders
        axis.axvline(mean - confint_n_std * std, ymin=zerolvl, color=cs[i],
                     alpha=1, linestyle='-', zorder=j * 101)
        axis.axvline(mean + confint_n_std * std, ymin=zerolvl, color=cs[i],
                     alpha=1, linestyle='-', zorder=j * 101)

        # Plot `ticks` number of xticks
        axis.xaxis.set_major_locator(ticker.MaxNLocator(ticks))

        legend_elements += [
            Line2D([0], [0], color=cs[i], linestyle='--', lw=3, label=f'({names[i]}) mean = {mean:.3f}±{confint_n_std * std:.3f}'),
            Patch(facecolor=colors['span'], alpha=1, edgecolor=cs[i], lw=2,
                  label=f'({names[i]}) std*{confint_n_std:.1f} = ({mean - confint_n_std * std:.3f}, {mean + confint_n_std * std:.3f})'),
        ]

    n_columns = len(arrays)

    axis.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.35),
                fancybox=True, shadow=False, ncol=n_columns,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['labelsize']})
