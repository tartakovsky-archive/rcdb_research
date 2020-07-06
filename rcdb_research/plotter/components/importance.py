import numpy as np
from typing import Optional
from matplotlib import ticker

from .. import style
from ..primitives import bars_pn


def importance(means: np.ndarray,
               mins: Optional[np.ndarray] = None,
               maxes: Optional[np.ndarray] = None,
               labels: Optional[list] = None,
               threshold: float = 0,
               orientation: str = 'h',
               title: Optional[str] = 'Feature importance',
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               fig_kwargs: Optional[dict] = None,
               ax_kwargs: Optional[dict] = None,
               pos_bar_kwargs: Optional[dict] = None,
               neg_bar_kwargs: Optional[dict] = None,
               sort: bool = False,
               ax=None):
    fig_kwargs = {
        **style.fig_kwargs(
            figsize=(16, 6) if orientation == 'h' else (16, 12)
        ),
        **(fig_kwargs or {})
    }

    ax_kwargs = {
        **style.ax_kwargs(
            xtickrotation=90 if orientation == 'v' else 0,
            xlocator=ticker.MaxNLocator(20) if orientation == 'h' else None,
            ylocator=ticker.MaxNLocator(20) if orientation == 'v' else None,
        ),
        **(ax_kwargs or {})
    }

    x = np.arange(means.shape[0])

    if sort:
        # --- Refactor?
        lists = [means, labels if labels is not None else list(range(len(labels)))]
        if mins is not None and maxes is not None:
            lists.append(mins)
            lists.append(maxes)

        sorted_lists = list((list(t) for t in zip(*sorted(zip(*lists), reverse=True))))

        means, labels = np.array(sorted_lists[0]), sorted_lists[1]

        if mins is not None and maxes is not None:
            mins = np.array(sorted_lists[2])
            maxes = np.array(sorted_lists[3])
        # ---

    fig, axis = bars_pn(y=means,
                        x=x,
                        width=0.8,
                        threshold=threshold,
                        orientation=orientation,
                        thr_orientation='v' if orientation == 'h' else 'h',
                        title=title,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        fig_kwargs=fig_kwargs,
                        ax_kwargs=ax_kwargs,
                        pos_bar_kwargs=pos_bar_kwargs,
                        neg_bar_kwargs=neg_bar_kwargs,
                        ax=ax)

    axis.invert_yaxis()

    if mins is not None and maxes is not None:
        if orientation == 'h':
            axis.hlines(y=x, xmin=mins, xmax=maxes)
        else:
            axis.vlines(x=x, ymin=mins, ymax=maxes)

    if orientation == 'h':
        axis.set_yticks(x)
        if labels is not None:
            axis.set_yticklabels(labels)
    else:
        axis.set_xticks(x)
        if labels is not None:
            axis.set_xticklabels(labels)

    if ax is None:
        return fig, axis
