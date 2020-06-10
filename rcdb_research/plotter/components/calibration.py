import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional, Callable

from .. import style
from ..utils import configure_axis
from ...metrics import prediction as metrics

from sklearn.metrics import mean_squared_error as mse


def calibration(a_true: np.ndarray,
                a_pred: np.ndarray,
                b_true: Optional[np.ndarray] = None,
                b_pred: Optional[np.ndarray] = None,
                a_name: str = 'A',
                b_name: str = 'B',
                n_std: float = 3,
                raw: bool = False,
                score_fn: Callable = mse,
                score_name: str = 'MSE',
                title: Optional[str] = 'Probability calibration curve',
                xlabel: Optional[str] = 'True probas',
                ylabel: Optional[str] = 'Predicted probas',
                fig_kwargs: Optional[dict] = None,
                ax_kwargs: Optional[dict] = None,
                line_a_kwargs: Optional[dict] = None,
                line_b_kwargs: Optional[dict] = None,
                ax=None) -> Optional[tuple]:
    fig_kwargs = {**style.fig_kwargs(figsize=(16, 7)), **(fig_kwargs or {})}
    ax_kwargs = {
        **style.ax_kwargs(
            xlocator=ticker.MaxNLocator(10),
            ylocator=ticker.MaxNLocator(10)
        ),
        **(ax_kwargs or {})
    }
    line_a_kwargs = {**style.line_kwargs(color='#49b4f2'), **(line_a_kwargs or {})}
    line_b_kwargs = {**style.line_kwargs(color='#f27549'), **(line_b_kwargs or {})}

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    legend_elements = []
    y_trues = [a_true]
    y_preds = [a_pred]
    names = [a_name]
    line_kwargs = [line_a_kwargs]

    if b_true is not None and b_pred is not None:
        y_trues.append(b_true)
        y_preds.append(b_pred)
        names.append(b_name)
        line_kwargs.append(line_b_kwargs)

    for i in range(len(y_trues)):
        if raw:
            axis.scatter(y_trues[i], y_preds[i], alpha=0.5, **line_kwargs[i])
            legend_elements += [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=line_kwargs[i]['color'],
                       markersize=10, label=names[i])
            ]
        else:
            true_probas, stds, pred_probas = metrics.calibration(y_trues[i], y_preds[i])
            score = score_fn(true_probas, pred_probas)

            axis.plot(true_probas, pred_probas, **line_kwargs[i])
            axis.fill_betweenx(pred_probas, true_probas - n_std * stds, true_probas + n_std * stds,
                               color=line_kwargs[i]['color'], alpha=0.5)

            legend_elements += [
                Patch(facecolor=line_kwargs[i]['color'],
                      edgecolor=line_kwargs[i]['color'],
                      label=f'{names[i]}, {score_name} = {score:.4f}', lw=2,
                      alpha=0.5)
            ]

    axis.plot([0, 1], [0, 1], "--", color='gray')

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.legend(handles=legend_elements, loc='upper left', fancybox=False,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['ticksize']})

    if ax is None:
        return fig, axis
