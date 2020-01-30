import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from copy import deepcopy
from typing import Union

from . import primitives
from .style import Style, colormap
from ..simulation.entities import Probabilities, Predictions, Trades

from sklearn.calibration import calibration_curve


def proba_report(probas: 'Probabilities', n_bins: int = 40, show_dates: bool = False):

    style = Style(fig_size=(16, 8))
    blue = colormap(0.56)

    fig = plt.figure(figsize=style.fig_size, facecolor="w", dpi=style.dpi)
    fig.suptitle("Probability report", x=0.526, y=1.05, fontsize=16)

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    primitives.curve(probas.y_pred_proba,
                     title='Probabilities over bars',
                     ylabel='Probability',
                     ax=ax1)

    if show_dates:
        primitives.second_index(
            ax1, _datestring(probas.index), xlabel='Bar number / Date', rotation=0,
        )

    ax2.plot([0, 1], [0, 1], "--", color='gray', label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = calibration_curve(
        probas.y_true, probas.y_pred_proba, normalize=False, n_bins=n_bins, strategy='uniform'
    )

    primitives.curve(y=fraction_of_positives, x=mean_predicted_value,
                     pos_color=blue, pos_legend_label='Predicted probas',
                     title='Probability calibration curve',
                     xlabel='Mean predicted probability', ylabel='Fraction of positives',
                     ax=ax2)

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.25))

    primitives.histogram(probas.y_pred_proba, n_bins=n_bins, n_ticks=20,
                         xlabel='Mean predicted probability', ylabel='Count', ax=ax3)

    fig.tight_layout(h_pad=4, w_pad=3)
    fig.show()


def preds_report(predictions: 'Predictions', window: int, threshold: float = 0.5,
                 show_dates: bool = False, target_label: Union[str, int] = 'all', neu_label: int = 0):

    style = Style(fig_size=(16, 6))
    blue = colormap(0.56)
    red = colormap(0.045)

    labels = dict(target_label=target_label, neu_label=neu_label)
    precision = predictions.precision(window=window, dense=True, **labels).fillna(threshold)

    h_pad = None
    if show_dates:
        style = Style(fig_size=(16, 7))
        h_pad = 4

    fig, axes = plt.subplots(2, 1,
                             figsize=style.fig_size, facecolor="w",
                             gridspec_kw={'height_ratios': [5, 2]}, dpi=style.dpi)

    fig.suptitle("Prediction report", x=0.528, y=1.05, fontsize=style.suptitle_size)

    precision_style = deepcopy(style)
    precision_style.fill = True
    precision_style.percent = True
    primitives.curve(precision,
                     threshold=threshold,
                     title='Rolling precision over bars',
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
                     title='Prediction density over bars',
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

    fig.suptitle("Trading simulation report", x=0.528, y=1.05, fontsize=style.suptitle_size)

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
                     title='Cumulative return over bars',
                     ylabel='Gain',
                     style=percent_fill_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax0)

    legend_elements = [
        Line2D([0], [0], lw=1, color=blue, label='Expectancy'),
        Patch(edgecolor=blue, facecolor=blue, alpha=0.7, label='Simulation')
    ]

    ax0.legend(handles=legend_elements, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.065))

    primitives.curve(trades.drawdown(**equity_params),
                     ylabel='Drawdown',
                     style=percent_fill_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax1)

    if show_dates:
        primitives.second_index(ax1, _datestring(trades.cum_return(**equity_params).index))

    primitives.curve(trades.returns(after_fees=after_fees, dense=dense),
                     ylabel='Returns',
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
