import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional

from .. import style
from ..utils import configure_axis


def calibration_colors(a='#49b4f2',
                       b='#f27549') -> dict:
    return locals()


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
                colors: Optional[dict] = None,
                fig_kwargs: Optional[dict] = None,
                ax_kwargs: Optional[dict] = None,
                line_kwargs: Optional[dict] = None,
                ax=None) -> Optional[tuple]:
    colors = colors or calibration_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs(figsize=(16, 7))
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        xlocator=ticker.MaxNLocator(10),
        ylocator=ticker.MaxNLocator(10)
    )
    line_kwargs = line_kwargs or style.line_kwargs(linewidth=2)
    _ = line_kwargs.pop('color', None)

    # Configure axis. Set labels, fonts, formatters, grid, etc.
    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    legend_elements = []
    y_trues = [a_true]
    y_preds = [a_pred]
    names = [a_name]
    cs = [colors['a']]

    if b_true is not None and b_pred is not None:
        y_trues.append(b_true)
        y_preds.append(b_pred)
        names.append(b_name)
        cs.append(colors['b'])

    for i in range(len(y_trues)):
        if raw:
            axis.scatter(y_trues[i], y_preds[i], color=cs[i], alpha=0.5)
            legend_elements += [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=cs[i], markersize=10, label=names[i])
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

            axis.plot(means, preds, color=cs[i], **line_kwargs)
            axis.fill_betweenx(preds, means - n_std * stds, means + n_std * stds, color=cs[i], alpha=0.5)

            legend_elements += [
                Patch(facecolor=cs[i], alpha=0.5, edgecolor=cs[i], label=names[i], lw=2)
            ]

    axis.plot([0, 1], [0, 1], "--", color='gray')

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    axis.legend(handles=legend_elements, loc='upper center',
                bbox_to_anchor=(0.5, 0.0),
                borderaxespad=-2,
                fancybox=True, shadow=False, ncol=2,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['labelsize']})

    if ax is None:
        return fig, axis
