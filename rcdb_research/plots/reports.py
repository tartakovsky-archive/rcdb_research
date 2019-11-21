from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import primitives
from .style import Style, ColorMap


def _datestring(index_array):
    return [d.strftime('%Y-%m-%d') for d in index_array]


def cv_report(cvresult, window, threshold, show_dates=False, style=Style(), colors=ColorMap()):
    precision = cvresult.precision(window=window, sparse=False).fillna(threshold)

    fig_size = style.fig_size
    h_pad = None
    if show_dates:
        fig_size = (fig_size[0], fig_size[1]+1)
        h_pad = 3

    fig, axes = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [5, 2]}, dpi=style.dpi)

    precision_style = deepcopy(style)
    precision_style.fill = True
    precision_style.percent = True
    primitives.curve(precision,
                     threshold=threshold,
                     title='Cross validation report',
                     xlabel=None if show_dates else 'Bar number',
                     ylabel=f'Precision, window={window}',
                     style=precision_style,
                     colors=colors,
                     ax=axes[0])
    if show_dates:
        primitives.second_index(axes[0], _datestring(precision.index))

    primitives.curve(cvresult.tp(),
                     style=style,
                     colors=colors,
                     ax=axes[1])

    primitives.curve(cvresult.fp()*-1,
                     xlabel=None if show_dates else 'Bar number',
                     ylabel='FP / TP',
                     style=style,
                     colors=colors,
                     ax=axes[1])
    if show_dates:
        primitives.second_index(axes[1], _datestring(cvresult.tp().index), xlabel='Bar number / Date')

    plt.tight_layout(h_pad=h_pad)
    plt.show()


def trading_report(analysis, show_dates=False, style=Style(), colors=ColorMap()):

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=style.fig_size,
        gridspec_kw={'height_ratios': [3, 1, 1]}, dpi=style.dpi
    )

    percent_style = deepcopy(style)
    percent_style.percent = True

    percent_fill_style = deepcopy(style)
    percent_fill_style.percent = True
    percent_fill_style.fill = True

    primitives.curve(analysis.expected_cum_return,
                     style=percent_style,
                     colors=colors,
                     ax=ax0)

    cr_style = deepcopy(style)
    cr_style.percent = True
    cr_style.fill = True
    primitives.curve(analysis.cum_return,
                     title='Trading simulation report',
                     ylabel='Gain',
                     style=percent_fill_style,
                     colors=colors,
                     ax=ax0)

    legend_elements = [
        Line2D([0], [0], lw=1, color=colors.positive, label='Expectancy'),
        Patch(edgecolor=colors.positive, facecolor=colors.positive, alpha=0.7, label='Simulation')
    ]

    ax0.legend(handles=legend_elements, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.058))

    primitives.curve(analysis.drawdown,
                     ylabel='Drawdown, %',
                     style=percent_fill_style,
                     colors=colors,
                     ax=ax1)

    if show_dates:
        primitives.add_second_index(ax1, _datestring(analysis.cum_return.index))

    primitives.curve(analysis.returns,
                     ylabel='Returns, %',
                     style=percent_style,
                     colors=colors,
                     ax=ax2)

    if show_dates:
        primitives.add_second_index(ax2, _datestring(analysis.cum_return.index), xlabel='Bar number / Date')

    plt.tight_layout()
    plt.show()
