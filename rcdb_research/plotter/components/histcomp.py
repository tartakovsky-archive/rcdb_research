import numpy as np

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch

from .. import style
from ..utils import configure_axis


def histcomp_colors(a='#49b4f2', b='#f27549') -> dict:
    return locals()


# Distribution Comparison Report
def histcomp(a: np.ndarray,
             b: Optional[np.ndarray] = None,
             a_name: str = '1',
             b_name: str = '2',
             bins: int = 20,
             ticks: int = 20,
             title: Optional[str] = 'Distributions of variables',
             xlabel: Optional[str] = 'Datapoints',
             ylabel: Optional[str] = 'Fraction',
             colors: Optional[dict] = None,
             fig_kwargs: Optional[dict] = None,
             ax_kwargs: Optional[dict] = None,
             hist_kwargs: Optional[dict] = None,
             ax=None) -> Optional[tuple]:
    colors = colors or histcomp_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 7))
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        tickrotation=45,
        xformatter=ticker.FormatStrFormatter('%.3f'),
        yformatter=ticker.FormatStrFormatter('%.2f'),
    )
    hist_kwargs = hist_kwargs or dict(alpha=0.85)
    _ = hist_kwargs.pop('color', None)
    _ = hist_kwargs.pop('weights', None)

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    legend_elements = []
    arrays = [a]
    names = [a_name]
    cs = [colors['a']]
    if b is not None:
        arrays.append(b)
        names.append(b_name)
        cs.append(colors['b'])

    # plot histograms
    for i, array in enumerate(arrays):
        axis.hist(array, bins=bins, color=cs[i], weights=np.ones(array.size) / array.size, **hist_kwargs)

        legend_elements += [
            Patch(facecolor=cs[i], alpha=1, label=names[i])
        ]

    # Plot `ticks` number of xticks
    axis.xaxis.set_major_locator(ticker.MaxNLocator(ticks))

    axis.legend(handles=legend_elements, loc='upper left',
                # bbox_to_anchor=(0.5, 1.0),
                # borderaxespad=2,
                fancybox=False, shadow=False, ncol=1,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['ticksize']})

    if ax is None:
        return fig, axis
