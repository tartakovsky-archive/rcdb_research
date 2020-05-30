import numpy as np

from typing import Optional
from matplotlib import ticker

from .. import style

from .. import primitives as prim


def hist_pn(y: np.array,
            bins: int = 20,
            ticks: int = 20,
            width: float = 1.0,
            threshold: float = 0,
            orientation: str = 'v',
            thr_orientation: str = 'v',
            title: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            fig_kwargs: Optional[dict] = None,
            ax_kwargs: Optional[dict] = None,
            pos_bar_kwargs: Optional[dict] = None,
            neg_bar_kwargs: Optional[dict] = None,
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

    return prim.bars_pn(x=center,
                        y=hist,
                        threshold=threshold,
                        width=bar_width,
                        orientation=orientation,
                        thr_orientation=thr_orientation,
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        fig_kwargs=fig_kwargs,
                        ax_kwargs=ax_kwargs,
                        pos_bar_kwargs=pos_bar_kwargs,
                        neg_bar_kwargs=neg_bar_kwargs,
                        ax=ax)
