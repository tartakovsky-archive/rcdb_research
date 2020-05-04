import logging
from functools import partial
from typing import List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.axes import Axes
from sklearn.model_selection import BaseCrossValidator, KFold

from .. import style
from .. import utils
from ...cross_validation import CombinatorialKFold, predicts_to_paths


def splits_colors(tainted='#414BB2',
                  train='#2D9BF0',
                  test='#FAC710',
                  embargo='#F24726',
                  gap='lightcoral') -> dict:
    return locals()


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
        show_groups: bool = False,
        show_paths: bool = False,
        ax: Optional[Axes] = None):
    colors = colors or splits_colors()
    fig_kwargs = fig_kwargs or style.fig_kwargs()
    ax_kwargs = ax_kwargs or style.ax_kwargs(
        xformatter=ticker.FormatStrFormatter('%.0f'),
        yformatter=ticker.FormatStrFormatter('%.0f'),
        xlocator=ticker.MaxNLocator(18, integer=True),
    )

    x_index = list(range(X.index.size))
    splits = list(cv.split(X=X, y=y))
    index = list(range(1, len(splits) + 1))

    # calculate starts & sizes of train & test bars for each split
    train_start, test_start, train_size, test_size = get_train_test(splits)

    # calculate starts & sizes of embargo bars
    if hasattr(cv, 'embargo') and cv.embargo:
        embargo_start, embargo_size = get_custom_bars(get_embargo_start, splits, cv.embargo, X)
    else:
        embargo_start, embargo_size = [], []

    # calculate starts & sizes of gap bars
    if hasattr(cv, 'last_n_gap_size') and cv.last_n_gap_size:
        gap_start, gap_size = get_custom_bars(get_gap_start, splits, cv.last_n_gap_size, X)
    else:
        gap_start, gap_size = [], []

    # calculate starts & sizes of tainted bars
    if hasattr(cv, 'tainted_up_to') and hasattr(cv, 'split_by_index') and cv.tainted_up_to:
        tainted_start, tainted_size = get_tainted(X, cv, index)
    else:
        tainted_start, tainted_size = [], []

    # calculate xs & ys of paths
    paths = []
    if show_paths:
        if hasattr(cv, 'n_folds') and hasattr(cv, 'k_tests'):
            paths = get_paths(X, cv)
        else:
            print('Warning: CV doesn`t support parameter `show_paths`')

    # calculate ys of groups
    groups = []
    if show_groups:
        if hasattr(cv, 'n_folds') or isinstance(cv, KFold):
            groups = get_groups(cv, X, train_start, test_start, tainted_size[0] if len(tainted_size) else None)
        else:
            print('Warning: CV doesn`t support parameter `show_groups`')

    fig, axis = (None, ax) if ax is not None else plt.subplots(**fig_kwargs)
    utils.configure_axis(axis, title, None if show_dates else xlabel, ylabel, ax_kwargs=ax_kwargs)

    # Y Axis
    if len(index) == 1:
        loc = ticker.MultipleLocator(base=5)
    else:
        loc = ticker.MaxNLocator(integer=True)

    axis.yaxis.set_major_locator(loc)
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda v, num: int(len(index) - v + 1)
        )
    )

    axis.set_ylim(index[0] - 0.5, index[-1] + 0.5)
    axis.set_xlim(x_index[0], x_index[-1] + 1)

    # Draw
    _draw_barh = partial(draw_barh, splits=splits, axis=axis)

    _draw_barh(train_start, train_size, 'train', colors['train'])
    _draw_barh(test_start, test_size, 'test', colors['test'])
    _draw_barh(embargo_start, embargo_size, 'embargo', colors['embargo'])
    _draw_barh(gap_start, gap_size, 'gap', colors['gap'])
    _draw_barh(tainted_start, tainted_size, 'tainted', colors['tainted'])

    if paths:
        draw_paths(paths, index, axis)

    if groups:
        draw_groups(groups, index, axis)

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
    idxs = np.where(np.hstack(([-1], np.diff(a))) != 1)[0]

    starts = a[idxs]
    sizes = [end_i - i for i, end_i in zip(idxs, idxs[1:].tolist() + [len(a)])]

    return starts, sizes


def get_embargo_start(train: np.ndarray, test: np.ndarray, size: int, *args, **kwargs) -> Optional[List[int]]:
    embargo_start = []

    for test_end in map(lambda b: sum(b) - 1, zip(*pieces(test))):
        for train_start in pieces(train)[0]:
            if train_start - test_end == size + 1:
                embargo_start.append(test_end + 1)

    return embargo_start


def get_gap_start(train: np.ndarray, test: np.ndarray, *args, **kwargs) -> Optional[List[int]]:
    test_start = test[0]
    if test_start != 0:
        train_before_test = train[train <= test_start]
        train_before_test_start = train_before_test[-1] if len(train_before_test) else None

        if train_before_test_start is not None and test_start - train_before_test_start != 1:
            return [train_before_test_start + 1]

    return None


def get_custom_bars(
    func: Callable,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    size: int,
    *args, **kwargs
) -> Tuple[List[List[int]], List[List[int]]]:
    starts = []
    sizes = []
    for train, test in splits:
        if len(train) and len(test):
            start = func(train, test, size=size)
            if start is not None:
                starts.append(start)
                sizes.append(np.repeat(size, len(start)).tolist())
                continue

        starts.append([0])
        sizes.append([0])

    return starts[::-1], sizes[::-1]


def get_paths(X: pd.DataFrame, cv: CombinatorialKFold) -> List[List[Tuple[int, int, int]]]:
    predicts_like = [
        {'idxs': test, 'split': np.repeat(split, test.shape)}
        for split, (_, test) in enumerate(cv.split(X))
    ]

    paths: List[List[Tuple[int, int, int]]] = []  # [[(y, start, width)..]..]
    for p in predicts_to_paths(predicts_like, k_tests=cv.k_tests, n_folds=cv.n_folds):
        path = []
        idxs = p['idxs']
        split = p['split']

        for s in np.unique(split):
            split_group = idxs[split == s]
            path.append(
                (s, split_group[0], len(split_group))
            )
        paths.append(path)
    return paths


def get_groups(cv, X, train_start, test_start, tainted_size=0):
    n_folds = getattr(cv, 'n_folds', None) or getattr(cv, 'n_splits')

    if not tainted_size:
        group_start = min(
            [
                min(np.hstack(train_start)),
                min(np.hstack(test_start))
            ]
        )
    else:
        group_start = tainted_size

    group_sizes = list(map(len, np.array_split(np.arange(len(X[group_start:])), n_folds)))
    return [group_start] + (group_start + np.cumsum(group_sizes)).tolist()[:-1]


def get_tainted(X, cv, index):
    # cv.split_by_index works only with pandaslike objects
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X_tainted, _ = cv.split_by_index(X, cv.tainted_up_to)
    tainted_start = np.repeat([0], len(index))
    tainted_size = np.repeat([len(X_tainted)], len(index))
    return tainted_start, tainted_size


def get_train_test(splits):
    train_start, test_start, train_size, test_size = [], [], [], []
    for i, (train, test) in reversed(list(enumerate(splits))):
        logging.debug(f'{i} train {train[:1]} {train[-1:]} test {test[:1]} {test[-1:]}')
        for dataset, starts, sizes in zip((train, test), (train_start, test_start), (train_size, test_size)):
            bars = pieces(dataset)
            starts.append(bars[0])
            sizes.append(bars[1])

    return train_start, test_start, train_size, test_size


def draw_barh(starts, sizes, label, color, axis, splits):
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


def draw_paths(paths, index, axis):
    cmap = plt.get_cmap('tab20_r')
    colors = np.linspace(0, 1, 20)

    for i, path in enumerate(paths):
        c = cmap(colors[i % colors.size])
        for y, start, width in path:
            axis.barh(y=len(index) - y, height=0.75, width=width, left=start, color=c)


def draw_groups(groups: List[int], index: List, axis):
    for i, g in enumerate(groups):
        axis.axvline(x=g, ymin=-1, ymax=2, c='black', linestyle='-', lw=0.5)
        axis.text(g - 0.3, len(index) + 0.7, str(i + 1))
