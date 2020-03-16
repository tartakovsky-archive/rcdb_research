import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

from copy import deepcopy
from typing import Union

from . import primitives
from .style import Style, colormap
from ..simulation import Probabilities, Predictions, Trades
from ..simulation import PredictionSimulator
    

def proba_report(probas: Probabilities, n_bins: int = 40, show_dates: bool = False):
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

    fraction_of_positives, mean_predicted_value = probas.metrics.calibration(
        normalize=False, n_bins=n_bins, strategy='uniform'
    )

    primitives.curve(y=fraction_of_positives, x=mean_predicted_value,
                     pos_color=blue, pos_legend_label='Predicted probas',
                     title='Probability calibration curve',
                     xlabel='Mean predicted probability', ylabel='Fraction of positives',
                     ax=ax2)

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.25))

    primitives.histogram(probas.y_pred_proba, n_bins=n_bins, n_ticks=20,
                         xlabel='Mean predicted probability', ylabel='Count', ax=ax3)

    # fig.tight_layout(h_pad=4, w_pad=3)
    fig.show()


def preds_report(preds: Predictions, window: int, threshold: float = 0.5, show_dates: bool = False):
    style = Style(fig_size=(16, 6))
    blue = colormap(0.56)
    red = colormap(0.045)

    precision = preds.metrics.precision(window=window, dense=True).fillna(threshold)

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

    primitives.curve(preds.metrics.tp(),
                     style=style,
                     pos_color=blue,
                     neg_color=red,
                     ax=axes[1])

    primitives.curve(preds.metrics.fp() * -1,
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
            _datestring(preds.metrics.tp().index), xlabel='Bar number / Date'
        )

    # plt.tight_layout(h_pad=h_pad)
    plt.show()


def trading_report(trades: Trades, show_dates: bool = False):
    style = Style(fig_size=(16, 9))
    blue = colormap(0.56)
    red = colormap(0.045)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1,
                                        figsize=style.fig_size, facecolor="w",
                                        gridspec_kw={'height_ratios': [3, 1, 1]}, dpi=style.dpi)

    fig.suptitle("Trading simulation report", x=0.528, y=1.05, fontsize=style.suptitle_size)

    percent_style = deepcopy(style)
    percent_style.percent = True

    percent_fill_style = deepcopy(style)
    percent_fill_style.percent = True
    percent_fill_style.fill = True

    primitives.curve(trades.metrics.cum_return(),
                     title='Cumulative return over bars',
                     ylabel='Gain',
                     style=percent_fill_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax0)

    primitives.curve(trades.metrics.drawdown(),
                     ylabel='Drawdown',
                     style=percent_fill_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax1)

    if show_dates:
        primitives.second_index(ax1, _datestring(trades.index))

    primitives.curve(trades.metrics.returns(),
                     ylabel='Returns',
                     style=percent_style,
                     pos_color=blue,
                     neg_color=red,
                     ax=ax2)

    if show_dates:
        primitives.second_index(
            ax2,
            _datestring(trades.index),
            xlabel='Bar number / Date'
        )

    # plt.tight_layout()
    plt.show()


def threshold_report(probas: Probabilities, activity_range: tuple = (0.05, 0.6),
                     n_steps: int = 40, direction: str = 'pos', tolerance: float = 1e-5):
    # Calculate threshold range, predictions and activities arrays
    max_threshold = probas.metrics.threshold_for_activity(activity_range[0], direction, tolerance)
    min_threshold = probas.metrics.threshold_for_activity(activity_range[1], direction, tolerance)
    thresholds = np.linspace(min_threshold, max_threshold, n_steps)

    preds_arr = [PredictionSimulator.preds(probas, t, direction).init_metrics(direction) for t in thresholds]

    precisions = np.array([p.metrics.precision() for p in preds_arr])
    activities = np.array([p.metrics.activity() for p in preds_arr])

    x_labels = [f"{t:.3f}" for t in thresholds]
    x_ticks = np.array((range(len(x_labels))))

    style = Style(fig_size=(16, 6))
    blue = colormap(0.56)
    orange = colormap(0.045)

    fig, ax = plt.subplots(1, 1, figsize=style.fig_size, facecolor="w", dpi=style.dpi)
    fig.suptitle("Threshold report", x=0.5, y=1.05, fontsize=style.suptitle_size)

    formatter = ticker.FormatStrFormatter('%.3f')
    ax2 = ax.twinx()

    primitives.curve(precisions, pos_color=orange, pos_legend_label='Precision',
                     xlabel='Threshold', ylabel='Precision', style=Style(line_width=2, marker='.'), ax=ax)
    primitives.curve(activities, pos_color=blue, pos_legend_label='Activity',
                     xlabel='Threshold', ylabel='Activity', style=Style(line_width=2, marker='.'), ax=ax2)

    p_min, p_max = precisions.min(), precisions.max()
    p_range = p_max - p_min
    a_min, a_max = activities.min(), activities.max()
    a_range = a_max - a_min
    # x_min, x_max = x_ticks.min(), x_ticks.max()
    # x_range = x_max - x_min

    for (x, p) in zip(x_ticks, precisions):
        if x % 3 == 0:
            ax.annotate(f'{p:.3f}', xy=(x - 0.6, p + p_range * 0.03), color=orange, fontweight='bold')

    for (x, a) in zip(x_ticks, activities):
        if x % 3 == 0:
            ax2.annotate(f'{a:.3f}', xy=(x - 0.6, a + a_range * 0.03), color=blue, fontweight='bold')

    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(ticker.LinearLocator(20))
    ax2.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_locator(ticker.LinearLocator(20))

    # Setup axis spines
    ax.set_frame_on(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    [i.set_linewidth(2.5) for i in ax.spines.values()]

    ax.spines['left'].set_color(orange)
    ax.spines['right'].set_color(blue)

    [t.set_color(orange) for t in ax.yaxis.get_ticklines()]
    [t.set_color(blue) for t in ax2.yaxis.get_ticklines()]

    fig.legend(loc='lower center', bbox_to_anchor=(0.495, 0.2))

    plt.xticks(ticks=x_ticks, labels=x_labels)

    [lbl.set_rotation(45) for lbl in ax.get_xticklabels()]

    # plt.tight_layout()
    plt.show()



#######
# Utility functions
#######


def _datestring(index_array: np.array):
    return [d.strftime('%Y-%m-%d') for d in index_array]
