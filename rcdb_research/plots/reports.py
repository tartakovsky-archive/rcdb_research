from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import primitives
from .style import Style, colormap

from ..metrics import Predictions, Trades

from typing import Union


def cv_report(predictions: 'Predictions', window: int, threshold: float = 0.5,
              show_dates: bool = False, label: Union[str, int] = 'all', neg_label: int = 0):

    style = Style(fig_size=(16, 6))
    blue = colormap(0.56)
    red = colormap(0.045)

    labels = dict(label=label, neg_label=neg_label)
    precision = predictions.precision(window=window, dense=True, **labels).fillna(threshold)

    h_pad = None
    if show_dates:
        style = Style(fig_size=(16, 7))
        h_pad = 3

    fig, axes = plt.subplots(2, 1,
                             figsize=style.fig_size, facecolor="w",
                             gridspec_kw={'height_ratios': [5, 2]}, dpi=style.dpi)

    precision_style = deepcopy(style)
    precision_style.fill = True
    precision_style.percent = True
    primitives.curve(precision,
                     threshold=threshold,
                     title='Cross validation report',
                     xlabel=None if show_dates else 'Bar number',
                     ylabel=f'Precision, window={window}',
                     style=precision_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=axes[0])
    if show_dates:
        primitives.second_index(axes[0], _datestring(precision.index))

    primitives.curve(predictions.tp(**labels),
                     style=style,
                     pos_color=blue,
                     neg_color=red,
                     ax=axes[1])

    primitives.curve(predictions.fp(**labels)*-1,
                     xlabel=None if show_dates else 'Bar number',
                     ylabel='FP / TP',
                     style=style,
                     pos_color=blue,
                     neg_color=red,
                     ax=axes[1])
    if show_dates:
        primitives.second_index(
            axes[1],
            _datestring(predictions.tp(**labels).index), xlabel='Bar number / Date'
        )

    plt.tight_layout(h_pad=h_pad)
    plt.show()


def trading_report(trades: 'Trades', show_dates: bool = False, initial: int = 100, position_size: float = 0.8,
                   after_fees: bool = True, dense: bool = False, compounded: bool = False):

    style = Style(fig_size=(16, 9))
    blue = colormap(0.56)
    red = colormap(0.045)

    equity_params = dict(initial=initial,
                         position_size=position_size,
                         after_fees=after_fees,
                         dense=dense,
                         compounded=compounded)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1,
                                        figsize=style.fig_size, facecolor="w",
                                        gridspec_kw={'height_ratios': [3, 1, 1]}, dpi=style.dpi)

    percent_style = deepcopy(style)
    percent_style.percent = True

    percent_fill_style = deepcopy(style)
    percent_fill_style.percent = True
    percent_fill_style.fill = True

    primitives.curve(trades.expected_cum_return(**equity_params),
                     style=percent_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax0)

    cr_style = deepcopy(style)
    cr_style.percent = True
    cr_style.fill = True
    primitives.curve(trades.cum_return(**equity_params),
                     title='Trading simulation report',
                     ylabel='Gain',
                     style=percent_fill_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax0)

    legend_elements = [
        Line2D([0], [0], lw=1, color=blue, label='Expectancy'),
        Patch(edgecolor=blue, facecolor=blue, alpha=0.7, label='Simulation')
    ]

    ax0.legend(handles=legend_elements, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.058))

    primitives.curve(trades.drawdown(**equity_params),
                     ylabel='Drawdown, %',
                     style=percent_fill_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax1)

    if show_dates:
        primitives.second_index(ax1, _datestring(trades.cum_return(**equity_params).index))

    primitives.curve(trades.returns(after_fees=after_fees, dense=dense),
                     ylabel='Returns, %',
                     style=percent_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax2)

    if show_dates:
        primitives.second_index(ax2, _datestring(trades.cum_return(**equity_params).index), xlabel='Bar number / Date')

    plt.tight_layout()
    plt.show()


#######
# Utility functions
#######


def _datestring(index_array: np.array):
    return [d.strftime('%Y-%m-%d') for d in index_array]
