import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import logging
from typing import Optional

from .. import primitives

from .. import style
from .. import utils


def pieces(a):
    idxs = np.argwhere(np.diff(a) > 1)

    if not len(idxs):
        return None, None

    idxs_after = np.hstack(idxs) + 1

    starts = [a[0]]
    sizes = [idxs_after[0]]

    for item, item_next in zip(idxs_after, idxs_after[1:]):
        starts.append(a[item])
        sizes.append(item_next - item + 1)

    starts.append(a[idxs_after[-1]])
    sizes.append(a[-1] - a[idxs_after[-1]] + 1)

    return starts, sizes


def get_embargo_start(train, test):
    test_end = test[-1]

    if test_end != len(X) - 1:
        train_after_test = train[train >= test_end]
        train_after_test_start = train_after_test[0] if len(train_after_test) else None

        if train_after_test_start is not None and train_after_test_start - test_end != 1:
            return test_end + 1

    return None


def get_gap_start(train, test):
    test_start = test[0]
    if test_start != 0:
        train_before_test = train[train <= test_start]
        train_before_test_start = train_before_test[-1] if len(train_before_test) else None

        if train_before_test_start is not None and test_start - train_before_test_start != 1:
            return train_before_test_start + 1

    return None


def get_custom_bars(func, splits, size):
    starts = []
    sizes = []
    for train, test in splits:
        if len(train) and len(test):
            start = func(train, test)
            if start is not None:
                starts.append(start)
                sizes.append(size)
                continue

        starts.append(0)
        sizes.append(0)

    return starts, sizes


def get_dataset_bars(dataset):
    parts = pieces(dataset)

    if len(dataset):
        if parts != (None, None):
            return parts
        else:
            return [dataset[0]], [dataset[-1] - dataset[0] + 1]

    return [], []


def plot_splits(cv, X, y=None, ax=None, colors=None):
    if colors is None:
        colors = dict(
            tainted='#414BB2',
            train='#2D9BF0',
            test='#FAC710',
            embargo='#F24726',
            gap='lightcoral',
        )

    train_start = []
    train_size = []
    test_start = []
    test_size = []

    embargo_start = []
    embargo_size = []

    gap_start = []
    gap_size = []

    tainted_start = []
    tainted_size = []

    splits = list(cv.split(X=X, y=y))
    index = list(range(1, len(splits) + 1))

    for i, (train, test) in enumerate(splits):
        logging.debug(f'{i} train {train[:1]} {train[-1:]} test {test[:1]} {test[-1:]}')
        for dataset, starts, sizes in zip((train, test), (train_start, test_start), (train_size, test_size)):
            bars = get_dataset_bars(dataset)
            starts.append(bars[0])
            sizes.append(bars[1])

    if hasattr(cv, 'embargo') and cv.embargo:
        embargo_start, embargo_size = get_custom_bars(get_embargo_start, splits, cv.embargo)

    if hasattr(cv, 'last_n_gap_size') and cv.last_n_gap_size:
        gap_start, gap_size = get_custom_bars(get_gap_start, splits, cv.last_n_gap_size)

    if hasattr(cv, 'tainted_up_to') and hasattr(cv, 'split_by_index') and cv.tainted_up_to:
        _X = X
        if not isinstance(X, pd.DataFrame):
            _X = pd.DataFrame(X)

        X_tainted, _ = cv.split_by_index(_X, cv.tainted_up_to)
        tainted_start = np.repeat(0, len(index))
        tainted_size = np.repeat(len(X_tainted), len(index))

    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=150) if ax is None else (None, ax)

    # Background
    ax1.set_frame_on(False)
    #     ax1.grid(color='lightgray', linestyle='-.', linewidth=0.5)
    ax1.grid(False)

    # Title
    ax1.set_title('CV splits over number of bars')
    # Legend
    # Y Axis
    if len(index) == 1:
        loc = ticker.MultipleLocator(base=5)
    else:
        loc = ticker.MaxNLocator(integer=True)

    ax1.yaxis.set_major_locator(loc)
    #     ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
    ax1.set_ylim(index[0] - 0.5, index[-1] + 0.5)
    ax1.set_ylabel('Split number', fontsize=12, labelpad=15)
    # X Axis
    ax1.set_xlabel('Bars', fontsize=12, labelpad=15)

    # Bars

    def draw_barh(starts, sizes, label, color):
        if not (len(sizes) and any(sizes)):
            return

        logging.debug(label)
        for i in index:
            logging.debug(f'{i} {starts[i - 1]} {sizes[i - 1]}')

            if hasattr(starts[i - 1], '__len__'):
                for start, size in zip(starts[i - 1], sizes[i - 1]):
                    ax1.barh(y=i, height=0.75, width=size, left=start, label=label, color=color)
                    label = ''
            else:
                ax1.barh(y=i, height=0.75, width=sizes[i - 1], left=starts[i - 1], label=label, color=color)
                label = ''

    draw_barh(train_start, train_size, 'train', colors['train'])
    draw_barh(test_start, test_size, 'test', colors['test'])
    draw_barh(embargo_start, embargo_size, 'embargo', colors['embargo'])
    draw_barh(gap_start, gap_size, 'gap', colors['gap'])
    draw_barh(tainted_start, tainted_size, 'tainted', colors['tainted'])

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncol=5)

    if ax is None:
        plt.tight_layout()
        plt.show()
