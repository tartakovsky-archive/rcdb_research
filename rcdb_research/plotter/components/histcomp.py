import numpy as np

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch

from .. import style
from ..utils import configure_axis


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
             fig_kwargs: Optional[dict] = None,
             ax_kwargs: Optional[dict] = None,
             hist_a_kwargs: Optional[dict] = None,
             hist_b_kwargs: Optional[dict] = None,
             ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(figsize=(16, 7)), **(fig_kwargs or {})}
    ax_kwargs = {
        **style.ax_kwargs(
            xtickrotation=45,
            xformatter=ticker.FormatStrFormatter('%.3f'),
            yformatter=ticker.FormatStrFormatter('%.2f'),
        ),
        **(ax_kwargs or {})
    }

    hist_a_kwargs = {**style.hist_kwargs(color='#49b4f2', alpha=0.85), **(hist_a_kwargs or {})}
    _ = hist_a_kwargs.pop('weights', None)

    hist_b_kwargs = {**style.hist_kwargs(color='#f27549', alpha=0.85), **(hist_b_kwargs or {})}
    _ = hist_b_kwargs.pop('weights', None)

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    legend_elements = []
    arrays = [a]
    names = [a_name]
    hist_kwargs = [hist_a_kwargs]

    if b is not None:
        arrays.append(b)
        names.append(b_name)
        hist_kwargs.append(hist_b_kwargs)

    # plot histograms
    for i, array in enumerate(arrays):
        axis.hist(array, bins=bins, weights=np.ones(array.size) / array.size, **hist_kwargs[i])

        legend_elements += [
            Patch(facecolor=hist_kwargs[i]['color'], alpha=1, label=names[i])
        ]

    # Plot `ticks` number of xticks
    axis.xaxis.set_major_locator(ticker.MaxNLocator(ticks))

    axis.legend(handles=legend_elements, loc='upper left',
                fancybox=False, shadow=False, ncol=1,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['ticksize']})

    if ax is None:
        return fig, axis
