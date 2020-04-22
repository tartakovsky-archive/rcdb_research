import matplotlib.pyplot as plt
from matplotlib import ticker

from typing import Optional

from .. import primitives
from ...simulation import Predictions

from .. import style
from .. import utils


def preds_colors(pos: str = '#49b4f2', neg: str = '#f27549') -> dict:
    return locals()


def preds_report(preds: Predictions, window: int, threshold: float = 0.5, show_dates: bool = False,
                 colors: Optional[dict] = None,
                 fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
                 line_kwargs: Optional[dict] = None):
    colors = colors or preds_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        xformatter=ticker.FormatStrFormatter('%.0f'),
    )
    line_kwargs = line_kwargs or style.line_kwargs(linewidth=1)

    precision = preds.metrics.precision(window=window, dense=True).fillna(threshold)

    if show_dates:
        fig_kwargs = {**fig_kwargs, 'figsize': (16, 7)}

    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 2]}, **fig_kwargs)

    fig.suptitle("Prediction report", x=0.528, y=1.05, **style.suptitle_kwargs())

    primitives.curve(precision,
                     threshold=threshold,
                     title='Rolling precision over bars',
                     xlabel=None if show_dates else 'Bar number',
                     ylabel=f'Precision, window={window}',
                     colors=colors,
                     ax_kwargs=ax_kwargs,
                     line_kwargs=line_kwargs,
                     fill=True,
                     ax=axes[0])

    if show_dates:
        utils.second_index(axes[0], utils.datestring(precision.index), ax_kwargs=ax_kwargs)

    primitives.line(preds.metrics.tp(),
                    colors={'main': colors['pos']},
                    ax_kwargs=ax_kwargs,
                    line_kwargs=line_kwargs,
                    ax=axes[1])

    primitives.line(preds.metrics.fp() * -1,
                    title='Prediction density over bars',
                    xlabel=None if show_dates else 'Bar number',
                    ylabel='FP / TP',
                    colors={'main': colors['neg']},
                    ax_kwargs=ax_kwargs,
                    line_kwargs=line_kwargs,
                    ax=axes[1])
    if show_dates:
        utils.second_index(
            axes[1],
            utils.datestring(preds.metrics.tp().index),
            xlabel='Bar number / Date',
            ax_kwargs=ax_kwargs
        )
