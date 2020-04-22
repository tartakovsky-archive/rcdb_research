import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Optional

from .. import primitives
from ...simulation import Probabilities, PredictionSimulator

from .. import style


def threshold_colors(activity: str = '#49b4f2',
                     precision: str = '#f27549', ) -> dict:
    return locals()


def threshold_report(probas: Probabilities, activity_range: tuple = (0.05, 0.6),
                     n_steps: int = 40, direction: str = 'pos', tolerance: float = 1e-5,
                     colors: Optional[dict] = None, fig_kwargs: Optional[dict] = None,
                     ax_kwargs: Optional[dict] = None, line_kwargs: Optional[dict] = None,
                     ):
    colors = colors or threshold_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 6))
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        tickrotation=45,
        yformatter=ticker.FormatStrFormatter('%.3f'),
    )
    line_kwargs = line_kwargs or style.line_kwargs(marker='.')

    # Calculate threshold range, predictions and activities arrays
    max_threshold = probas.metrics.threshold_for_activity(activity_range[0], direction, tolerance)
    min_threshold = probas.metrics.threshold_for_activity(activity_range[1], direction, tolerance)
    thresholds = np.linspace(min_threshold, max_threshold, n_steps)

    preds_arr = [PredictionSimulator.preds(probas, t, direction).init_metrics(direction) for t in thresholds]

    precisions = np.array([p.metrics.precision() for p in preds_arr])
    activities = np.array([p.metrics.activity() for p in preds_arr])

    x_labels = [f"{t:.3f}" for t in thresholds]
    x_ticks = np.arange(len(x_labels))

    fig, ax = plt.subplots(1, 1, **fig_kwargs)
    fig.suptitle("Threshold report", x=0.5, y=1.1, **style.suptitle_kwargs())

    ax2 = ax.twinx()

    primitives.line(precisions,
                    xlabel='Threshold',
                    ylabel='Precision',
                    legend='Precision',
                    colors={'main': colors['precision']},
                    ax_kwargs={**ax_kwargs, 'ylocator': ticker.LinearLocator(20)},
                    line_kwargs=line_kwargs,
                    ax=ax)

    primitives.line(activities,
                    xlabel='Threshold',
                    ylabel='Activity',
                    legend='Activity',
                    colors={'main': colors['activity']},
                    ax_kwargs={**ax_kwargs, 'ylocator': ticker.LinearLocator(20)},
                    line_kwargs=line_kwargs,
                    ax=ax2)

    p_min, p_max = precisions.min(), precisions.max()
    p_range = p_max - p_min
    a_min, a_max = activities.min(), activities.max()
    a_range = a_max - a_min
    # x_min, x_max = x_ticks.min(), x_ticks.max()
    # x_range = x_max - x_min

    for (x, p) in zip(x_ticks, precisions):
        if x % 3 == 0:
            ax.annotate(f'{p:.3f}',
                        xy=(x - 0.6, p + p_range * 0.03),
                        color=colors['precision'],
                        fontweight='bold',
                        fontfamily=ax_kwargs['fontfamily'])

    for (x, a) in zip(x_ticks, activities):
        if x % 3 == 0:
            ax2.annotate(f'{a:.3f}',
                         xy=(x - 0.6, a + a_range * 0.03),
                         color=colors['activity'],
                         fontweight='bold',
                         fontfamily=ax_kwargs['fontfamily'])

    # Setup axis spines
    ax.set_frame_on(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    [i.set_linewidth(2.5) for i in ax.spines.values()]

    ax.spines['left'].set_color(colors['precision'])
    ax.spines['right'].set_color(colors['activity'])

    [t.set_color(colors['precision']) for t in ax.yaxis.get_ticklines()]
    [t.set_color(colors['activity']) for t in ax2.yaxis.get_ticklines()]

    fig.legend(loc='lower center', bbox_to_anchor=(0.495, 0.2))

    plt.xticks(ticks=x_ticks, labels=x_labels)
