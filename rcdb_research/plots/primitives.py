import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from . import palette

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

def curve(array, threshold=0, title=None, xlabel=None, ylabel=None,
          pos_color=palette.blue, neg_color=palette.red,
          fill=False, percent=False, showx=True, figsize=(16, 4), ax=None):

    x = list(range(array.size))
    y = array

    if percent:
        y = y*100
        threshold = threshold*100

    fig, ax1 = plt.subplots(figsize=figsize) if ax is None else (None, ax)

    ax1.set_frame_on(False)
    ax1.grid(color='lightgray', linestyle='-.', linewidth=0.5)

    formatter = ticker.PercentFormatter(decimals=0) if percent else ticker.FormatStrFormatter('%.2f')
    ax1.yaxis.set_major_formatter(formatter)
    if not showx:
        ax1.xaxis.set_major_formatter(ticker.NullFormatter())

    ax1.set_title(title)
    ax1.set_xlabel(xlabel, fontsize=12, labelpad=15)
    ax1.set_ylabel(ylabel, fontsize=12, labelpad=15)

    if np.where(y >= threshold)[0].size > 0:
        y_pos = np.where(y >= threshold, y, np.nan)
        y_pos[_edges_of_nans(y_pos)] = threshold
        ax1.plot(x, y_pos, color=pos_color, linewidth=1)

    if np.where(y <= threshold)[0].size > 0:
        y_neg = np.where(y <= threshold, y, np.nan)
        y_neg[_edges_of_nans(y_neg)] = threshold
        ax1.plot(x, y_neg, color=neg_color, linewidth=1)

    if fill:
        ax1.fill_between(x, threshold, y, where=(y >= threshold), facecolor=pos_color, alpha=0.7)
        ax1.fill_between(x, threshold, y, where=(y <= threshold), facecolor=neg_color, alpha=0.7)

    if ax is None:
        plt.tight_layout()
        plt.show()

def splits(cv, X, y=None, train_color=palette.blue, test_color=palette.red, ax=None):

    train_start = []
    train_size = []
    test_start = []
    test_size = []
    for i, (train, test) in enumerate(cv.split(X=X, y=y)):
        train_start.append(train[0])
        train_size.append(train[-1]-train[0])
        test_start.append(test[0])
        test_size.append(test[-1]-test[0])

    index = list(range(1, len(train_start) + 1))

    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=150) if ax is None else (None, ax)

    # Background
    ax1.set_frame_on(False)
    ax1.grid(color='lightgray', linestyle='-.', linewidth=0.5)

    # Title
    ax1.set_title('CV splits over number of bars')

    # Y Axis
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
    ax1.set_ylim(index[0]-0.5, index[-1]+0.5)
    ax1.set_ylabel('Split number', fontsize=12, labelpad=15)

    # X Axis
    ax1.set_xlabel('Bars', fontsize=12, labelpad=15)

    # Bars
    ax1.barh(y=index, height=0.75, width=train_size, left=train_start, label='Train set', color=train_color)
    ax1.barh(y=index, height=0.75, width=test_size, left=test_start, label='Test set', color=test_color)

    # Legend
    ax1.legend(loc='lower right')

    if ax is None:
        plt.tight_layout()
        plt.show()

def histogram(array, nbins=100, nticks=50, color=palette.blue, ax=None):
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=100) if ax is None else (None, ax)

    hist, bins = np.histogram(array, bins=nbins)
    width = 0.75 * (bins[1] - bins[0])
    x = (bins[:-1] + bins[1:]) / 2

    # Background
    ax1.set_frame_on(False)
    ax1.grid(color='lightgray', linestyle='-.', linewidth=0.5)

    # Title
    ax1.set_title('Histogram')

    # Y Axis
    ax1.set_ylabel('Number of occurences', fontsize=12, labelpad=15)

    # X Axis
    ax1.set_xlabel('Value bins', fontsize=12, labelpad=15)
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nticks))
    plt.xticks(rotation=50)

    # Bars
    ax1.bar(x=x, height=hist, width=width, color=color)

    if ax is None:
        plt.tight_layout()
        plt.show()

def add_second_index(ax, x2, xlabel=None, rotation=0):
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
