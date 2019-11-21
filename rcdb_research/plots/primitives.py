import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .style import Style, ColorMap


def curve(array, threshold=0, title=None, xlabel=None, ylabel=None,
          style=Style(), colors=ColorMap(), ax=None):

    x = list(range(array.size))
    y = array

    if style.percent:
        y = y*100
        threshold = threshold*100

    fig, ax1 = plt.subplots(figsize=style.fig_size, dpi=style.dpi, facecolor="w") if ax is None else (None, ax)
    _configure_axis(ax1, style)

    ax1.set_title(title)
    ax1.set_xlabel(xlabel, fontsize=12, labelpad=15)
    ax1.set_ylabel(ylabel, fontsize=12, labelpad=15)

    y_pos = np.where(y >= threshold, y, np.nan)
    y_neg = np.where(y < threshold, y, np.nan)

    y_pos[_edges_of_nans(y_pos)] = threshold
    y_neg[_edges_of_nans(y_neg)] = threshold

    ax1.plot(x, y_pos, color=colors.positive, linewidth=1)
    ax1.plot(x, y_neg, color=colors.negative, linewidth=1)

    if style.fill:
        ax1.fill_between(x, threshold, y_pos, facecolor=colors.positive, alpha=0.7)
        ax1.fill_between(x, threshold, y_neg, facecolor=colors.negative, alpha=0.7)

    if ax is None:
        plt.tight_layout()
        plt.show()


def splits(cv, X, y=None, style=Style(), colors=ColorMap(), ax=None):

    train_start = []
    train_size = []
    test_start = []
    test_size = []
    for i, (train, test) in enumerate(cv.split(X=X, y=y)):
        train_start.append(train[0])
        train_size.append(train[-1]-train[0]+1)
        test_start.append(test[0])
        test_size.append(test[-1]-test[0]+1)

    index = list(range(1, len(train_start) + 1))

    fig, ax1 = plt.subplots(figsize=style.fig_size, dpi=style.dpi, facecolor="w") if ax is None else (None, ax)
    _configure_axis(ax1, style)

    # Title
    ax1.set_title('CV splits over number of bars')

    # Y Axis
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
    ax1.set_ylim(index[0]-0.5, index[-1]+0.5)
    ax1.set_ylabel('Split number', fontsize=12, labelpad=15)

    # X Axis
    ax1.set_xlabel('Bars', fontsize=12, labelpad=15)

    # Bars
    ax1.barh(y=index, height=0.75, width=train_size, left=train_start,
             label='Train set', color=colors.train_set)
    ax1.barh(y=index, height=0.75, width=test_size, left=test_start,
             label='Test set', color=colors.test_set)

    # Legend
    ax1.legend(loc='lower right')

    if ax is None:
        plt.tight_layout()
        plt.show()


def histogram(array, nbins=100, nticks=50, style=Style(), colors=ColorMap(), ax=None):

    fig, ax1 = plt.subplots(figsize=style.fig_size, dpi=style.dpi, facecolor="w") if ax is None else (None, ax)
    _configure_axis(ax1, style)

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
    ax1.bar(x=x, height=hist, width=width, color=colors.positive)

    if ax is None:
        plt.tight_layout()
        plt.show()


def bars(array, threshold=0, title=None, xlabel=None, ylabel=None, style=Style(), colors=ColorMap(), ax=None):

    x = list(range(array.size))
    y = array

    if style.percent:
        y = y*100
        threshold = threshold*100

    fig, ax1 = plt.subplots(figsize=style.fig_size, facecolor="w") if ax is None else (None, ax)
    _configure_axis(ax1, style)

    ax1.set_title(title)
    ax1.set_xlabel(xlabel, fontsize=style.label_size, labelpad=15)
    ax1.set_ylabel(ylabel, fontsize=style.label_size, labelpad=15)

    if np.where(y >= threshold)[0].size > 0:
        y_pos = np.where(y >= threshold, y-threshold, np.nan)
        ax1.bar(x, height=y_pos, bottom=threshold, color=colors.positive, linewidth=1)

    if np.where(y <= threshold)[0].size > 0:
        y_neg = np.where(y <= threshold, y-threshold, np.nan)
        ax1.bar(x, height=y_neg, bottom=threshold, color=colors.negative, linewidth=1)

    if ax is None:
        plt.tight_layout()
        plt.show()


def area(array, array2, title=None, xlabel=None, ylabel=None, style=Style(), colors=ColorMap(), ax=None):

    x = list(range(array.size))
    y = array
    y2 = array2

    if style.percent:
        y = y*100
        y2 = y2*100

    fig, ax1 = plt.subplots(figsize=style.fig_size, facecolor="w") if ax is None else (None, ax)
    _configure_axis(ax1, style)

    ax1.set_title(title)
    ax1.set_xlabel(xlabel, fontsize=style.label_size, labelpad=15)
    ax1.set_ylabel(ylabel, fontsize=style.label_size, labelpad=15)

    ax1.plot(x, y, color=colors.positive, linewidth=1)
    ax1.plot(x, y2, color=colors.positive, linewidth=1)
    ax1.fill_between(x, y, y2, facecolor=colors.positive, alpha=style.fill_alpha)

    if ax is None:
        plt.tight_layout()
        plt.show()


def second_index(ax, x2, xlabel=None, rotation=0):
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

#######
# Utility functions
#######


def _configure_axis(ax, style):
    ax.set_frame_on(False)
    ax.grid(color='lightgray', linestyle='-.', linewidth=0.5)

    formatter = ticker.PercentFormatter(decimals=0) if style.percent else ticker.FormatStrFormatter('%.2f')
    ax.yaxis.set_major_formatter(formatter)
    if not style.show_x:
        ax.xaxis.set_major_formatter(ticker.NullFormatter())
    if not style.show_y:
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.tick_params(axis='both', which='major', labelsize=style.tick_size)


def _edges_of_nans(array):
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
