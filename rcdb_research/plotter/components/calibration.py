import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional

from .. import style
from ..utils import configure_axis


def calibration(a_true: np.ndarray,
                a_pred: np.ndarray,
                b_true: Optional[np.ndarray] = None,
                b_pred: Optional[np.ndarray] = None,
                a_name: str = 'A',
                b_name: str = 'B',
                n_std: float = 3,
                raw: bool = False,
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
                Line2D([0], [0], marker='o', color='w', markerfacecolor=line_kwargs[i]['color'], markersize=10, label=names[i])
            ]
        else:
            bins = np.linspace(0, 1, 25)
            binids = np.digitize(y_preds[i], bins) - 1

            means = []
            stds = []
            preds = []
            for binid in np.unique(binids):
                select = binids == binid
                data = np.hstack(y_trues[i][select])
                means.append(data.mean())
                stds.append(data.std())
                preds.append(y_preds[i][select].mean())
            means = np.array(means)
            stds = np.array(stds)
            preds = np.array(preds)

            axis.plot(means, preds, **line_kwargs[i])
            axis.fill_betweenx(preds, means - n_std * stds, means + n_std * stds,
                               color=line_kwargs[i]['color'], alpha=0.5)

            legend_elements += [
                Patch(facecolor=line_kwargs[i]['color'], alpha=0.5,
                      edgecolor=line_kwargs[i]['color'], label=names[i], lw=2)
            ]

    axis.plot([0, 1], [0, 1], "--", color='gray')

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.legend(handles=legend_elements, loc='upper left', fancybox=False,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['ticksize']})

    if ax is None:
        return fig, axis
