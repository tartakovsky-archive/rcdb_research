import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from . import primitives
from . import palette

def _datestring(index_array):
    return [d.strftime('%Y-%m-%d') for d in index_array]

def cv_report(cvresult, window, threshold, show_dates=False, dpi=150):
    precision = cvresult.precision(window=window, sparse=False).fillna(threshold)

    fig_h = 7 if show_dates else 6
    h_pad = 3 if show_dates else None

    fig, axes = plt.subplots(2, 1, figsize=(16, fig_h), gridspec_kw={'height_ratios': [5, 2]}, dpi=dpi)

    primitives.curve(precision,
                     threshold=threshold,
                     title='Cross validation report',
                     xlabel=None if show_dates else 'Bar number',
                     ylabel=f'Precision, window={window}',
                     fill=True,
                     percent=True,
                     pos_color=palette.blue,
                     neg_color=palette.red,
                     ax=axes[0])
    if show_dates:
        primitives.add_second_index(axes[0], _datestring(precision.index))

    primitives.curve(cvresult.tp(),
                     pos_color=palette.blue,
                     neg_color=palette.red,
                     ax=axes[1])


    primitives.curve(cvresult.fp()*-1,
                     xlabel=None if show_dates else 'Bar number',
                     ylabel='FP / TP',
                     pos_color=palette.blue,
                     neg_color=palette.red,
                     ax=axes[1])
    if show_dates:
        primitives.add_second_index(axes[1], _datestring(cvresult.tp().index), xlabel='Bar number / Date')

    plt.tight_layout(h_pad=h_pad)
    plt.show()


def trading_report(analysis, show_dates=False, dpi=150):

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3, 1, 1]}, dpi=dpi)

    primitives.curve(analysis.expected_cum_return,
                     percent=True,
                     pos_color=palette.blue,
                     neg_color=palette.red,
                     ax=ax0)

    primitives.curve(analysis.cum_return,
                     title='Trading simulation report',
                     ylabel='Gain',
                     percent=True,
                     fill=True,
                     pos_color=palette.blue,
                     neg_color=palette.red,
                     ax=ax0)

    legend_elements = [
        Line2D([0], [0], lw=1, color=palette.blue, label='Expectancy'),
        Patch(edgecolor=palette.blue, facecolor=palette.blue, alpha=0.7, label='Simulation')
    ]

    ax0.legend(handles=legend_elements, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.058))

    primitives.curve(analysis.drawdown,
                     ylabel='Drawdown, %',
                     percent=True,
                     fill=True,
                     pos_color=palette.blue,
                     neg_color=palette.red,
                     ax=ax1)

    if show_dates:
        primitives.add_second_index(ax1, _datestring(analysis.cum_return.index))

    primitives.curve(analysis.returns,
                     ylabel='Returns, %',
                     percent=True,
                     pos_color=palette.blue,
                     neg_color=palette.red,
                     ax=ax2)

    if show_dates:
        primitives.add_second_index(ax2, _datestring(analysis.cum_return.index), xlabel='Bar number / Date')

    plt.tight_layout()
    plt.show()
