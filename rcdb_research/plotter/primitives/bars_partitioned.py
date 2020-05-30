import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from typing import Optional, Tuple

from .. import style
from ..utils import configure_axis


def bars_partitioned(lengths: np.array,
                     positions: np.array = None,
                     thresholds: Tuple[float] = (),
                     bar_kwargs: Tuple[dict] = (),
                     width: float = 1.0,
                     ticks: int = 20,
                     orientation: str = 'v',
                     title: Optional[str] = None,
                     xlabel: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     fig_kwargs: Optional[dict] = None,
                     ax_kwargs: Optional[dict] = None,
                     ax=None) -> Optional[tuple]:
    supported_orientations = ['v', 'h', 'vertical', 'horizontal']
    if orientation not in supported_orientations:
        raise ValueError(
            f'{orientation} orientation is not supported. Should be one of the following: {supported_orientations}'
        )

    if orientation == 'v':
        orientation = 'vertical'
    elif orientation == 'h':
        orientation = 'horizontal'

    fig_kwargs = {**style.fig_kwargs(), **(fig_kwargs or {})}
    ax_kwargs = {
        **style.ax_kwargs(
            tickrotation=45,
            xlocator=ticker.MaxNLocator(ticks) if orientation == 'vertical' else None,
            ylocator=ticker.MaxNLocator(ticks) if orientation == 'horizontal' else None,
        ),
        **(ax_kwargs or {})
    }

    default_bar_kwargs = [
        dict(color='#f27549'),
        dict(color='#49b4f2'),
        dict(color='#4ECF64'),
    ]

    if len(bar_kwargs) == 0:
        # Setup up to three default bar styles if user haven't provided any
        bar_kwargs = default_bar_kwargs[:len(thresholds) + 1]

    if len(thresholds) > 0 and not len(bar_kwargs) == len(thresholds) + 1:
        raise ValueError(
            f'If thresholds are passed there should be len(thresholds) dicts in bar_kwargs tuple.'
            f'The function can provide up to three default bar_kwargs if user has not passed any.'
            f'Looks like there was not enough bar_kwargs to style all partitions.'
            f'len(bar_kwargs) = {len(bar_kwargs)}, instead of expected {len(thresholds) + 1}'
        )

    positions = np.arange(y.size) if positions is None else positions

    fig, axis = plt.subplots(**fig_kwargs) if ax is None else (plt.gcf(), ax)
    configure_axis(axis, title, xlabel, ylabel, ax_kwargs=ax_kwargs)

    partitions = []
    for i in range(len(thresholds)):
        if i == 0:
            # first threshold
            partitions.append({
                'position': positions[positions < thresholds[i]],
                'length': lengths[positions < thresholds[i]]
            })
        if i != 0:
            # middle thresholds
            partitions.append({
                'position': positions[(positions >= thresholds[i - 1]) & (positions < thresholds[i])],
                'length': lengths[(positions >= thresholds[i - 1]) & (positions < thresholds[i])]
            })
        if i == len(thresholds) - 1:
            # last threshold
            partitions.append({
                'position': positions[positions > thresholds[i]],
                'length': lengths[positions > thresholds[i]]
            })

    if len(partitions) == 0:
        partitions.append({'position': positions, 'length': lengths})

    for p, kws in zip(partitions, bar_kwargs):
        if orientation == 'vertical':
            axis.bar(x=p['position'], height=p['length'], width=width, **kws)
        else:
            axis.barh(y=p['position'], width=p['length'], height=width, **kws)

    for t in thresholds:
        if orientation == 'vertical':
            axis.axvline(x=t, linewidth=1, linestyle='--', color='black')
        else:
            axis.axhline(y=t, linewidth=1, linestyle='--', color='black')

    if ax is None:
        return fig, axis
