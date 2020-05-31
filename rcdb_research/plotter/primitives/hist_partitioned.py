import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import ticker

from typing import Optional, Tuple

from .. import style
from .. import primitives as prim


def hist_partitioned(y: np.array,
                     bins: int = 20,
                     ticks: int = 20,
                     width: float = 1.0,
                     thresholds: Tuple[float] = (),
                     bar_kwargs: Tuple[dict] = (),
                     orientation: str = 'v',
                     plot_median: bool = True,
                     title: Optional[str] = None,
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     fig_kwargs: Optional[dict] = None,
                     ax_kwargs: Optional[dict] = None,
                     ax=None) -> Optional[tuple]:
    ax_kwargs = {
        **style.ax_kwargs(
            tickrotation=45,
            xlocator=ticker.MaxNLocator(ticks) if orientation == 'v' else None,
            ylocator=ticker.MaxNLocator(ticks) if orientation == 'h' else None,
        ),
        **(ax_kwargs or {})
    }

    hist, bin_edges = np.histogram(y, bins=bins, weights=np.ones(y.size) / y.size)
    bar_width = width * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2

    default_bar_kwargs = [
        dict(color='#f27549'),
        dict(color='#49b4f2'),
        dict(color='#4ECF64'),
    ]

    if len(bar_kwargs) == 0:
        # Setup up to three default bar styles if user haven't provided any
        bar_kwargs = default_bar_kwargs[:len(thresholds) + 1]

    out = prim.bars_partitioned(lengths=hist,
                                positions=center,
                                thresholds=thresholds,
                                bar_kwargs=bar_kwargs,
                                width=bar_width,
                                ticks=ticks,
                                orientation=orientation,
                                title=title,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                fig_kwargs=fig_kwargs,
                                ax_kwargs=ax_kwargs,
                                ax=ax)

    fig, axis = out if ax is None else (plt.gcf(), ax)

    median = np.median(y)
    if plot_median:
        if orientation == 'v':
            axis.axvline(x=median, linewidth=3, linestyle='--', color='#cccccc')
        elif orientation == 'h':
            axis.axhline(y=median, linewidth=3, linestyle='--', color='#cccccc')

    partitions = []
    for i in range(len(thresholds)):
        if i == 0:
            # first threshold
            partitions.append(y[y < thresholds[i]])
        if i != 0:
            # middle thresholds
            partitions.append(y[(y >= thresholds[i - i]) & (y < thresholds[i])])
        if i == len(thresholds) - 1:
            # last threshold
            partitions.append(y[y > thresholds[i]])

    fractions = [p.size / y.size for p in partitions]

    legend_elements = []
    for i, f in enumerate(fractions):
        if i != 0:
            legend_elements.append(
                Line2D([0], [0], color='black', linestyle='--', lw=1, label=f'{thresholds[i - 1]}'),
            )
        legend_elements.append(
            Patch(facecolor=bar_kwargs[i]['color'], label=f'{f * 100:.2f}%')
        )

    if plot_median:
        legend_elements.append(
            Line2D([0], [0], color='#cccccc', linestyle='--', lw=3, label=f'q.5 = {median:.2f}'),
        )

    axis.legend(handles=legend_elements, loc='best', fancybox=False,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['labelsize']})

    if ax is None:
        return fig, axis
