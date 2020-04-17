import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import Formatter

from . import style

from typing import Optional


def second_index(ax, x2: np.array, xlabel: Optional[str] = None, ax_kwargs=None):
    ax_kwargs = ax_kwargs or style.ax_kwargs()

    x1 = list(ax.lines[0].get_xdata())

    x1_tick_locs = ax.get_xticks()
    x1_tick_loc_ids = [(x1.index(loc) if loc in x1 else None) for loc in x1_tick_locs]
    x2_tick_labels = [(x2[i] if i is not None else None) for i in x1_tick_loc_ids]

    ax2 = ax.twiny()
    ax2.set_frame_on(False)
    ax2.set_xticks(x1_tick_locs)
    ax2.set_xticklabels(x2_tick_labels)
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 20))

    ax2.set_xlabel(xlabel, fontsize=ax_kwargs['labelsize'], labelpad=ax_kwargs['labelpad'])

    [lbl.set_rotation(ax_kwargs['tickrotation']) for lbl in ax2.get_xticklabels()]


def configure_axis(ax, title='', xlabel='', ylabel='',
                   xformatter: Formatter = ticker.FormatStrFormatter('%.2f'),
                   yformatter: Formatter = ticker.FormatStrFormatter('%.2f'),
                   ax_kwargs=None):
    ax_kwargs = ax_kwargs or style.ax_kwargs()

    ax.set_frame_on(False)
    ax.grid(color='lightgray', linestyle='-.', linewidth=0.5)
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    if not ax_kwargs['showx']:
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
    if not ax_kwargs['showy']:
        ax.yaxis.set_major_formatter(ticker.NullFormatter())

    ax.tick_params(axis='both', which='major', labelsize=ax_kwargs['ticksize'])
    ax.set_title(title, fontsize=ax_kwargs['titlesize'],
                 family=ax_kwargs['fontfamily'], pad=ax_kwargs['titlepad'])
    ax.set_xlabel(xlabel, fontsize=ax_kwargs['labelsize'],
                  labelpad=ax_kwargs['labelpad'], family=ax_kwargs['fontfamily'])
    ax.set_ylabel(ylabel, fontsize=ax_kwargs['labelsize'],
                  labelpad=ax_kwargs['labelpad'], family=ax_kwargs['fontfamily'])

    ax.tick_params(axis='both', which='major', labelsize=ax_kwargs['labelsize'])

    for tick in ax.get_xticklabels():
        tick.set_fontfamily(ax_kwargs['fontfamily'])
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(ax_kwargs['fontfamily'])

    [lbl.set_rotation(ax_kwargs['tickrotation']) for lbl in ax.get_xticklabels()]
