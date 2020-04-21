import logging
from typing import List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from sklearn.model_selection import BaseCrossValidator

from .. import style
from .. import utils


def splits_colors(tainted='#414BB2',
                  train='#2D9BF0',
                  test='#FAC710',
                  embargo='#F24726',
                  gap='lightcoral') -> dict: return locals()


def splits(
        cv: BaseCrossValidator,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        title: Optional[str] = 'CV splits over number of bars',
        xlabel: Optional[str] = 'Bar number',
        ylabel: Optional[str] = 'Split number',
        colors: Optional[dict] = None,
        fig_kwargs: Optional[dict] = None,
        ax_kwargs: Optional[dict] = None,
        show_dates: bool = False,
        ax: Optional[Axes] = None):
    colors = colors or splits_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        xformatter=ticker.FormatStrFormatter('%.0f'),
        yformatter=ticker.FormatStrFormatter('%.0f'),
        xlocator=ticker.MaxNLocator(18, integer=True),
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

    x_index = list(range(X.index.size))
    splits = list(cv.split(X=X, y=y))
    index = list(range(1, len(splits) + 1))

    # calculate starts & sizes of train & test bars for each split
    for i, (train, test) in enumerate(splits):
        logging.debug(f'{i} train {train[:1]} {train[-1:]} test {test[:1]} {test[-1:]}')
        for dataset, starts, sizes in zip((train, test), (train_start, test_start), (train_size, test_size)):
            bars = get_dataset_bars(dataset)
            starts.append(bars[0])
            sizes.append(bars[1])

    # calculate starts & sizes of embargo bars
    if hasattr(cv, 'embargo') and cv.embargo:
        embargo_start, embargo_size = get_custom_bars(get_embargo_start, splits, cv.embargo, X)

    # calculate starts & sizes of gap bars
    if hasattr(cv, 'last_n_gap_size') and cv.last_n_gap_size:
        gap_start, gap_size = get_custom_bars(get_gap_start, splits, cv.last_n_gap_size, X)

    # calculate starts & sizes of tainted bars
    if hasattr(cv, 'tainted_up_to') and hasattr(cv, 'split_by_index') and cv.tainted_up_to:
        _X = X

        # cv.split_by_index works only with pandaslike objects
        if not isinstance(X, pd.DataFrame):
            _X = pd.DataFrame(X)

        X_tainted, _ = cv.split_by_index(_X, cv.tainted_up_to)
        tainted_start = np.repeat(0, len(index))
        tainted_size = np.repeat(len(X_tainted), len(index))

    fig, axis = (None, ax) if ax is not None else plt.subplots(**fig_kwargs)
    utils.configure_axis(axis, title, None if show_dates else xlabel, ylabel, ax_kwargs=ax_kwargs)

    # Y Axis
    if len(index) == 1:
        loc = ticker.MultipleLocator(base=5)
    else:
        loc = ticker.MaxNLocator(integer=True)

    axis.yaxis.set_major_locator(loc)

    axis.set_ylim(index[0] - 0.5, index[-1] + 0.5)
    axis.set_xlim(x_index[0], x_index[-1])

    # Bars
    def draw_barh(starts, sizes, label, color):
        if not (len(sizes) and any(sizes)):
            return

        logging.debug(label)
        for i in range(len(splits)):
            y = i + 1
            logging.debug(f'{i} {starts[i]} {sizes[i]}')

            if hasattr(starts[i], '__len__'):
                for start, size in zip(starts[i], sizes[i]):
                    axis.barh(y=y, height=0.75, width=size, left=start, label=label, color=color)
                    label = ''  # fix duplicates in legend
            else:
                axis.barh(y=y, height=0.75, width=sizes[i], left=starts[i], label=label, color=color)
                label = ''  # fix duplicates in legend

    draw_barh(train_start, train_size, 'train', colors['train'])
    draw_barh(test_start, test_size, 'test', colors['test'])
    draw_barh(embargo_start, embargo_size, 'embargo', colors['embargo'])
    draw_barh(gap_start, gap_size, 'gap', colors['gap'])
    draw_barh(tainted_start, tainted_size, 'tainted', colors['tainted'])

    axis.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35),
                fancybox=True, shadow=False, ncol=5,
                prop={'family': ax_kwargs['fontfamily'], 'size': ax_kwargs['labelsize']})

    if show_dates:
        utils.second_index(axis,
                           x2=utils.datestring(X.index),
                           x1=x_index,
                           xlabel='Bar number / Date',
                           ax_kwargs={**ax_kwargs, 'tickrotation': 15})


#########################################
# Utility functions for the main report #
#########################################

def pieces(a: np.ndarray) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Split dataset idxs into bar pieces

    1,2,4,5,8 -> 1,2 | 4,5 | 8 -> bar starts: [1, 4, 8], bar sizes: [2, 3, 1]

    :param a: dataset indexes
    :return:
    """
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


def get_embargo_start(train: np.ndarray, test: np.ndarray, X: pd.DataFrame) -> Optional[int]:
    test_end = test[-1]

    if test_end != len(X) - 1:
        train_after_test = train[train >= test_end]
        train_after_test_start = train_after_test[0] if len(train_after_test) else None

        if train_after_test_start is not None and train_after_test_start - test_end != 1:
            return test_end + 1

    return None


def get_gap_start(train: np.ndarray, test: np.ndarray, *args) -> Optional[int]:
    test_start = test[0]
    if test_start != 0:
        train_before_test = train[train <= test_start]
        train_before_test_start = train_before_test[-1] if len(train_before_test) else None

        if train_before_test_start is not None and test_start - train_before_test_start != 1:
            return train_before_test_start + 1

    return None


def get_custom_bars(
        func: Callable,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        size: int,
        X: pd.DataFrame
) -> Tuple[List[int], List[int]]:
    starts = []
    sizes = []
    for train, test in splits:
        if len(train) and len(test):
            start = func(train, test, X)
            if start is not None:
                starts.append(start)
                sizes.append(size)
                continue

        starts.append(0)
        sizes.append(0)

    return starts, sizes


def get_dataset_bars(index: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Calculates bars start indexes & starts from dataset index
    :param index: dataset index
    :return: tuple of lists with start bar indexes and bar sizes (starts, sizes)
    """

    parts = pieces(index)

    if len(index):
        if parts != (None, None):
            return parts
        else:
            return [index[0]], [index[-1] - index[0] + 1]

    return [], []
