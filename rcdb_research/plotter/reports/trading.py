import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Optional

from .. import primitives
from ...simulation import Trades

from .. import style
from .. import utils


def trading_colors(pos: str = '#49b4f2',
                   neg: str = '#f27549') -> dict:
    return locals()


def trading_report(trades: Trades, show_dates: bool = False,
                   colors: Optional[dict] = None, fig_kwargs: Optional[dict] = None,
                   ax_kwargs: Optional[dict] = None, line_kwargs: Optional[dict] = None):
    colors = colors or trading_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 9))
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        xformatter=ticker.FormatStrFormatter('%.0f'),
        yformatter=ticker.PercentFormatter(xmax=1, decimals=0),
    )
    line_kwargs = line_kwargs or style.line_kwargs(linewidth=1)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]}, **fig_kwargs)

    fig.suptitle("Trading simulation report", x=0.528, y=1.05, **style.suptitle_kwargs())

    primitives.curve(trades.metrics.cum_return(),
                     title='Cumulative return over bars',
                     ylabel='Gain',
                     fill=True,
                     colors=colors,
                     ax_kwargs={**ax_kwargs, 'ylocator': ticker.MaxNLocator(12)},
                     line_kwargs=line_kwargs,
                     ax=ax0)

    primitives.line(trades.metrics.drawdown(),
                    ylabel='Drawdown',
                    fill=True,
                    colors={'main': colors['neg']},
                    ax_kwargs={**ax_kwargs, 'ylocator': ticker.MaxNLocator(5)},
                    line_kwargs=line_kwargs,
                    ax=ax1)

    if show_dates:
        utils.second_index(ax1, utils.datestring(trades.index), ax_kwargs=ax_kwargs)

    primitives.curve(trades.metrics.returns(),
                     ylabel='Returns',
                     colors=colors,
                     ax_kwargs={**ax_kwargs, 'ylocator': ticker.MaxNLocator(5)},
                     line_kwargs=line_kwargs,
                     ax=ax2)

    if show_dates:
        utils.second_index(
            ax2,
            utils.datestring(trades.index),
            xlabel='Bar number / Date',
            ax_kwargs=ax_kwargs
        )
