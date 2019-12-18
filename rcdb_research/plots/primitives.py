import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .style import Style, colormap

from typing import Optional


def curve(array: np.array, threshold: float = 0, title: Optional[str] = None,
          xlabel: Optional[str] = None, ylabel: Optional[str] = None,
          pos_legend_label: Optional[str] = None, neg_legend_label: Optional[str] = None,
          pos_color: tuple = colormap(0.56), neg_color: tuple = colormap(0.045),
          style: 'Style' = Style(), ax=None):

    x = list(range(array.size))
    y = array

    if style.percent:
        y = y*100
        threshold = threshold*100

    fig, ax1 = plt.subplots(figsize=style.fig_size, dpi=style.dpi, facecolor="w") if ax is None else (None, ax)
    configure_axis(ax1, style)

    ax1.set_title(title)
    ax1.set_xlabel(xlabel, fontsize=12, labelpad=15)
    ax1.set_ylabel(ylabel, fontsize=12, labelpad=15)

    y_pos = np.where(y >= threshold, y, np.nan)
    y_neg = np.where(y < threshold, y, np.nan)

    y_pos[_edges_of_nans(y_pos)] = threshold
    y_neg[_edges_of_nans(y_neg)] = threshold

    ax1.plot(x, y_pos, color=pos_color, linewidth=1, label=pos_legend_label)
    ax1.plot(x, y_neg, color=neg_color, linewidth=1, label=neg_legend_label)

    if style.fill:
        ax1.fill_between(x, threshold, y_pos, facecolor=pos_color, alpha=0.7)
        ax1.fill_between(x, threshold, y_neg, facecolor=neg_color, alpha=0.7)

    if ax is None:
        plt.tight_layout()
        plt.show()


def histogram(array: np.array, nbins: int = 100, nticks: int = 50,
              style: 'Style' = Style(), color: tuple = colormap(0.56), ax=None):

    fig, ax1 = plt.subplots(figsize=style.fig_size, dpi=style.dpi, facecolor="w") if ax is None else (None, ax)
    configure_axis(ax1, style)

    hist, bins = np.histogram(array, bins=nbins)
    width = 0.75 * (bins[1] - bins[0])
    x = (bins[:-1] + bins[1:]) / 2

    # Setup labels
    ax1.set_title('Histogram')
    ax1.set_ylabel('Number of occurences', fontsize=style.label_size, labelpad=15)
    ax1.set_xlabel('Value bins', fontsize=style.label_size, labelpad=15)

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nticks))
    plt.xticks(rotation=50)

    # Bars
    ax1.bar(x=x, height=hist, width=width, color=color)

    if ax is None:
        plt.tight_layout()
        plt.show()


def bars(array: np.array, threshold: float = 0, title: Optional[str] = None,
         xlabel: Optional[str] = None, ylabel: Optional[str] = None, style: 'Style' = Style(),
         pos_color: tuple = colormap(0.56), neg_color: tuple = colormap(0.045),
         pos_legend_label: Optional[str] = None, neg_legend_label: Optional[str] = None,
         ax=None):

    x = list(range(array.size))
    y = array

    if style.percent:
        y = y*100
        threshold = threshold*100

    fig, ax1 = plt.subplots(figsize=style.fig_size, facecolor="w") if ax is None else (None, ax)
    configure_axis(ax1, style)

    ax1.set_title(title)
    ax1.set_xlabel(xlabel, fontsize=style.label_size, labelpad=15)
    ax1.set_ylabel(ylabel, fontsize=style.label_size, labelpad=15)

    if np.where(y >= threshold)[0].size > 0:
        y_pos = np.where(y >= threshold, y-threshold, np.nan)
        ax1.bar(x, height=y_pos, bottom=threshold, color=pos_color, linewidth=1, label=pos_legend_label)

    if np.where(y <= threshold)[0].size > 0:
        y_neg = np.where(y <= threshold, y-threshold, np.nan)
        ax1.bar(x, height=y_neg, bottom=threshold, color=neg_color, linewidth=1, label=neg_legend_label)

    if ax is None:
        plt.tight_layout()
        plt.show()


def area(a1: np.array, a2: np.array, title: Optional[str] = None,
         xlabel: Optional[str] = None, ylabel: Optional[str] = None, style: 'Style' = Style(),
         color: tuple = colormap(0.56), legend_label: Optional[str] = None, ax=None):

    x = list(range(a1.size))
    y = a1
    y2 = a2

    if style.percent:
        y = y*100
        y2 = y2*100

    fig, ax1 = plt.subplots(figsize=style.fig_size, facecolor="w") if ax is None else (None, ax)
    configure_axis(ax1, style)

    ax1.set_title(title)
    ax1.set_xlabel(xlabel, fontsize=style.label_size, labelpad=15)
    ax1.set_ylabel(ylabel, fontsize=style.label_size, labelpad=15)

    ax1.plot(x, y, color=color, linewidth=1, label=legend_label)
    ax1.plot(x, y2, color=color, linewidth=1)
    ax1.fill_between(x, y, y2, facecolor=color, alpha=style.fill_alpha)

    if ax is None:
        plt.tight_layout()
        plt.show()


def cmap_gradient(cmap, n_segments: int = 101, n_ticks: int = 51):
    gradient = np.linspace(0.0, 1.0, n_segments)
    gradient_2d = np.vstack((gradient, gradient))

    fig, ax = plt.subplots(1, 1, figsize=(12, 2), dpi=100, facecolor="w")

    ticks = np.linspace(0.0, gradient.size-1, n_ticks)
    labels = [f'{x:.2f}' for x in np.linspace(0.0, 1.0, n_ticks)]

    ax.imshow(gradient_2d, aspect='auto', cmap=cmap)

    ax.yaxis.set_major_locator(ticker.NullLocator())
    plt.xticks(ticks=ticks, labels=labels, rotation=45)

    fig.tight_layout()
    fig.show()


def second_index(ax, x2: np.array, xlabel: Optional[str] = None, rotation: float = 0):
    x1 = list(ax.lines[0].get_xdata())

    x1_tick_locs = ax.get_xticks()
    x1_tick_loc_ids = [(x1.index(l) if l in x1 else None) for l in x1_tick_locs]
    x2_tick_labels = [(x2[i] if i is not None else None) for i in x1_tick_loc_ids]

    ax2 = ax.twiny()
    ax2.set_frame_on(False)
    ax2.set_xticks(x1_tick_locs)
    ax2.set_xticklabels(x2_tick_labels)
    ax2.set_xlim(ax.get_xlim())
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 20))

    ax2.set_xlabel(xlabel, fontsize=12, labelpad=15)

    plt.xticks(rotation=0)


def configure_axis(ax, style: 'Style'):
    ax.set_frame_on(False)
    ax.grid(color='lightgray', linestyle='-.', linewidth=0.5)

    formatter = ticker.PercentFormatter(decimals=0) if style.percent else ticker.FormatStrFormatter('%.2f')
    ax.yaxis.set_major_formatter(formatter)
    if not style.show_x:
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
    if not style.show_y:
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.tick_params(axis='both', which='major', labelsize=style.tick_size)

#######
# Utility functions
#######


def _edges_of_nans(array: np.array):
    # display(array)
    # > [1, nan, nan, 2, 3, nan, 1, nan, nan, nan]
    isnan = np.concatenate(([0], np.isnan(array), [0]))
    # > [0 0 1 1 0 0 1 0 1 1 1 0]
    changes = np.abs(np.diff(isnan))
    # > [0 1 0 1 0 1 1 1 0 0 1]
    ranges = np.where(changes == 1)[0].reshape(-1, 2)
    # > [[ 1  3], [ 5  6], [ 7 10]]
    ranges[:, 1] = ranges[:, 1] - 1
    # > [[1 2], [5 5], [7 9]]
    edges = np.unique(ranges.ravel())
    # > [1 2 5 7 9]
    return edges
