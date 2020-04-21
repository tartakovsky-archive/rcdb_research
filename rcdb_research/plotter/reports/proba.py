import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Optional

from .. import primitives
from ...simulation import Probabilities

from .. import style
from .. import utils


def proba_colors(probas: str = '#49b4f2',
                 calibration: str = '#49b4f2',
                 hist: str = '#49b4f2') -> dict: return locals()


def proba_report(probas: Probabilities, n_bins: int = 40, show_dates: bool = False, colors: Optional[dict] = None,
                 fig_kwargs: Optional[dict] = None, ax_kwargs: Optional[dict] = None,
                 line_kwargs: Optional[dict] = None, hist_kwargs: Optional[dict] = None) -> tuple:
    colors = colors or proba_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 9))
    ax_kwargs = ax_kwargs or style.ax_kwargs()
    line_kwargs = line_kwargs or style.line_kwargs(linewidth=1)

    fig = plt.figure(**fig_kwargs)
    fig.suptitle("Probability report", x=0.526, y=1.05, **style.suptitle_kwargs())

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    primitives.line(probas.y_pred_proba,
                    title='Probabilities over bars',
                    ylabel='Probability',
                    colors={'main': colors['probas']},
                    ax_kwargs={
                        **ax_kwargs,
                        'xformatter': ticker.FormatStrFormatter('%.0f')
                    },
                    line_kwargs=line_kwargs,
                    ax=ax1)

    if show_dates:
        utils.second_index(ax1, utils.datestring(probas.index),
                           xlabel='Bar number / Date', ax_kwargs=ax_kwargs)

    ax2.plot([0, 1], [0, 1], "--", color='gray', label="Perfectly calibrated")

    fraction_of_positives, mean_predicted_value = probas.metrics.calibration(
        normalize=False, n_bins=n_bins, strategy='uniform'
    )

    primitives.line(y=fraction_of_positives,
                    x=mean_predicted_value,
                    colors={'main': colors['calibration']},
                    legend='Predicted probas',
                    title='Probability calibration curve',
                    xlabel='Mean predicted probability',
                    ylabel='Fraction of positives',
                    ax_kwargs=ax_kwargs,
                    line_kwargs=line_kwargs,
                    ax=ax2)

    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.25))

    primitives.hist(probas.y_pred_proba, bins=n_bins, ticks=20,
                    xlabel='Mean predicted probability', ylabel='Count',
                    colors={'main': colors['hist']},
                    ax_kwargs={**ax_kwargs, 'tickrotation': 45},
                    hist_kwargs=hist_kwargs, ax=ax3)
