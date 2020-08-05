import copy
import inspect
from operator import itemgetter
from itertools import combinations, takewhile
from functools import reduce
from typing import Union, List, Tuple, Iterable, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ... import utils

PandasLike = Union[pd.DataFrame, pd.Series]
Split = Tuple[np.ndarray, np.ndarray]
Splits = List[Split]


def consistent_itemgetter(*items):
    if len(items) == 1:
        item = items[0]

        def g(obj):
            return (obj[item],)
    else:
        def g(obj):
            return tuple(obj[item] for item in items)
    return g


class CombinatorialCV(BaseCrossValidator):
    class SplitterException(Exception):
        pass

    def __init__(
            self,
            n_folds: int,
            embargo_bars: int = 0,
            embargo_pct: float = 0.,
            tainted_up_to: Optional[Any] = None,
            k_tests: int = 1
    ):
        if n_folds <= 1 and tainted_up_to is None:
            raise self.SplitterException(
                "No data left for training set. "
                "Either set n_folds to > 1 or mark some data as tainted by setting tainted_up_to to not None"
            )

        if n_folds <= k_tests and not (n_folds <= 1 and k_tests == 1):
            raise ValueError('n_folds value must be higher then k_tests')

        self.n_folds = n_folds
        self.embargo_bars = embargo_bars
        self.embargo_pct = embargo_pct
        self.tainted_up_to = tainted_up_to
        self.k_tests = k_tests

        self._embargo_sizes = []
        self._embargo_starts = []

    @staticmethod
    def get_n_paths(k_tests, n_folds):
        if k_tests == 1:
            return 1

        return reduce(
            lambda a, b: a * b,
            map(lambda i: n_folds - i, range(1, k_tests))
        ) // np.math.factorial(k_tests - 1)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return reduce(
            lambda a, b: a * b,
            map(lambda i: self.n_folds - i, range(self.k_tests))
        ) // np.math.factorial(self.k_tests)

    @staticmethod
    def split_by_index(data: PandasLike, index) -> Tuple[PandasLike, PandasLike]:
        return data[data.index <= index], data[data.index > index]

    def split_by_tainted(self, X: Union[PandasLike, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if self.tainted_up_to is not None:
            if not isinstance(X, pd.DataFrame) or not isinstance(X, pd.Series):
                X = pd.DataFrame(X)

            X_tainted, X = self.split_by_index(X, self.tainted_up_to)
            tainted_idxs = np.arange(0, len(X_tainted))
            idxs = np.arange(len(X_tainted), len(X_tainted) + len(X))

            if not len(idxs) or (self.n_folds <= 1 and not len(tainted_idxs)):
                raise self.SplitterException('No data left for test set after separating tainted observations')
        else:
            tainted_idxs = np.array([], dtype=np.int)
            idxs = np.arange(len(X))

        return tainted_idxs, idxs

    def split(self, X: Union[PandasLike, np.ndarray], *args, **kwargs) -> List[Split]:
        tainted_idxs, idxs = self.split_by_tainted(X)

        if self.n_folds <= 1 and self.k_tests == 1:
            return [(tainted_idxs, idxs)]

        if len(idxs) < self.n_folds:
            raise self.SplitterException(f'Not enough data left to perform {self.n_folds} folds')

        groups = np.array_split(idxs, self.n_folds)
        if self.embargo_bars and not all(len(g) - self.embargo_bars > 0 for g in groups[self.k_tests:]):
            raise self.SplitterException('Not enough train set data to embargo')

        groups_idxs = np.arange(self.n_folds)

        splits = []

        for test_folds in combinations(range(self.n_folds), self.k_tests):
            train_folds = groups_idxs[~np.isin(groups_idxs, test_folds)]
            train_groups = list(consistent_itemgetter(*train_folds)(groups))

            test_groups = list(consistent_itemgetter(*test_folds)(groups))

            if len(tainted_idxs):
                train_groups.insert(0, tainted_idxs)

            test_groups, train_groups = self.purge(test_groups, train_groups)

            if self.embargo_bars != 0 or self.embargo_pct != 0:
                test_groups, train_groups = self.apply_embargo(
                    test_groups,
                    train_groups,
                    test_folds,
                    train_folds
                )

            test = np.hstack(test_groups)
            train = np.hstack(train_groups)

            splits.append((train, test))

        return splits

    def purge(
            self,
            test_groups: List[np.ndarray],
            train_groups: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return test_groups, train_groups

    @staticmethod
    def find_consecutive_groups(groups: List[np.ndarray], folds: List[int], to_i: int) -> np.ndarray:
        """
        Example:

        groups = [[0, 1], [4, 5], [6, 7], [8, 9]]
                                     ^i == 2
        folds = [0, 2, 3, 4]
        to_i = 2

        res = [4, 5, 6, 7]

        :param groups:
        :param folds:
        :param to_i:
        :return:
        """
        res = []
        prev_fold = None
        for group, fold in reversed(list(zip(groups[:to_i + 1], folds[:to_i + 1]))):
            if prev_fold is None or prev_fold - fold == 1:
                res.append(group)
                prev_fold = fold
            else:
                break

        res = res[::-1]
        return np.hstack(res)

    @staticmethod
    def cut_embargo(train_folds, train_groups, train_i, embargo_size):
        embargo_size_rest = embargo_size - len(train_groups[train_i])
        last_fold_idx = train_folds[train_i]
        if len(train_groups) > train_i + 1:
            for i in range(train_i + 1, len(train_groups)):
                if train_folds[i] - last_fold_idx != 1:
                    break

                train_group = train_groups[i]
                train_groups[i] = train_group[embargo_size_rest:]
                if len(train_group) > embargo_size_rest:
                    embargo_size_rest = 0
                    break

                embargo_size_rest -= len(train_group)
                last_fold_idx = train_folds[i]

        return embargo_size - embargo_size_rest

    @staticmethod
    def _gen_test_train_series(split):
        pos = 0
        while pos < len(split):
            # take test groups series
            test_groups = []
            for is_train, fold, group in split[pos:]:
                if is_train:
                    break
                test_groups.append(group)
                pos += 1

            train_groups = []
            for is_train, fold, group in split[pos:]:
                if not is_train:
                    break
                train_groups.append(group)
                pos += 1

            yield test_groups, train_groups

    def apply_embargo(
            self,
            test_groups: List[np.ndarray],
            train_groups: List[np.ndarray],
            test_folds: List[int],
            train_folds: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        _test_groups = test_groups
        tainted = None
        if len(train_groups) != len(train_folds):  # tainted
            tainted = train_groups[0]
            train_groups = train_groups[1:]

        embargo_starts = []
        embargo_sizes = []
        new_train_groups = []

        # list with tuples [([test_fold0, test_fold1], [train_fold2, train_fold3]), ...]
        is_train = True
        _trains = [(is_train, fold, group) for fold, group in zip(train_folds, train_groups)]
        is_train = False
        _tests = [(is_train, fold, group) for fold, group in zip(test_folds, test_groups)]

        split = sorted(_trains + _tests, key=itemgetter(1))

        # cut leading trains

        cut = 0
        for is_train, fold, group in split:
            if not is_train:
                break
            new_train_groups.append(group)
            cut += 1

        if cut != len(train_groups):
            split = split[cut:]

            for test_groups, train_groups in self._gen_test_train_series(split):
                if not train_groups:
                    continue

                embargo_size = max(
                    (
                        int(sum(len(g) for g in test_groups) * self.embargo_pct),
                        self.embargo_bars
                    )
                )
                if not embargo_size:
                    continue

                embargo_size_initial = embargo_size

                embargo_starts.append(train_groups[0][0])
                for train_group in train_groups:
                    train_group_len = len(train_group)
                    new_train_groups.append(
                        train_group[embargo_size:]
                    )

                    if embargo_size >= train_group_len:
                        embargo_size -= train_group_len
                    else:
                        embargo_size = 0

                embargo_sizes.append(embargo_size_initial - embargo_size)

        if tainted is not None:
            new_train_groups.insert(0, tainted)

        self._embargo_sizes.append(embargo_sizes)
        self._embargo_starts.append(embargo_starts)

        return _test_groups, new_train_groups

    def _iter_test_indices(self, X=None, y=None, groups=None):
        return map(lambda split: split[1], self.split(X, y, groups))


class CombinatorialPurgedCV(CombinatorialCV):
    def __init__(
            self,
            n_folds: int,
            bars_timestamp_start: np.ndarray,
            bars_timestamp_end: np.ndarray,
            embargo_bars: int = 0,
            embargo_pct: float = 0.,
            tainted_up_to: Optional[Any] = None,
            k_tests: int = 1,
    ):
        super().__init__(n_folds, embargo_bars, embargo_pct, tainted_up_to, k_tests)

        self._purged_bars = []

        if bars_timestamp_start.shape != bars_timestamp_end.shape:
            raise ValueError('bars_timestamp_start and bars_timestamp_end has different shape')

        self.bars_timestamp_start = bars_timestamp_start
        self.bars_timestamp_end = bars_timestamp_end

        self.test_folds_sizes = []

    def split(self, X: Union[PandasLike, np.ndarray], *args, **kwargs) -> List[Split]:
        self.test_folds_sizes.clear()
        self._purged_bars.clear()
        return super().split(X, *args, **kwargs)

    def purge(
            self,
            test_groups: List[np.ndarray],
            train_groups: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.array]]:
        _test_groups = test_groups
        test_groups = copy.deepcopy(test_groups)
        groups_lasts = sorted(
            [
                [x[-1], False, i] for i, x in enumerate(test_groups)
            ] + [
                [x[-1], True, i] for i, x in enumerate(train_groups)
            ],
            key=itemgetter(0)
        )

        grouped_groups_lasts = [groups_lasts[0]]
        for idx, is_train, i in groups_lasts[1:]:
            latest_idx, is_latest_is_train, _ = grouped_groups_lasts[-1]
            if is_train and is_latest_is_train == is_train:
                grouped_groups_lasts[-1][0] = idx
                grouped_groups_lasts[-1][2] = i
            else:
                grouped_groups_lasts.append([idx, is_train, i])

        for i, (train_idx, *_) in filter(lambda x: x[1][1], enumerate(grouped_groups_lasts)):
            for *_, test_group_i in takewhile(lambda g: not g[1], grouped_groups_lasts[i + 1:]):
                test_group = test_groups[test_group_i]
                test_groups[test_group_i] = \
                    test_group[self.bars_timestamp_start[test_group] > self.bars_timestamp_end[train_idx]]
        self.test_folds_sizes.append(
            [len(g) for g in test_groups]
        )

        self._purged_bars.append(
            np.sort(
                list(set(np.hstack(_test_groups)) - set(np.hstack(test_groups)))
            )
        )
        return test_groups, train_groups


def split_indexes_to_bars(
        X: Union[PandasLike, np.ndarray],
        y: Union[PandasLike, np.ndarray],
        indexes: Iterable[Split],
        raw: bool = False
) -> List[Dict[str, Union[PandasLike, np.ndarray]]]:
    if isinstance(X, np.ndarray):
        def iloc(data, idxs):
            return data[idxs]
    elif raw:
        def iloc(data, idxs):
            return data.iloc[idxs].values
    else:
        def iloc(data, idxs):
            return data.iloc[idxs]

    return [
        {
            'X_train': iloc(X, train),
            'y_train': iloc(y, train),
            'X_test': iloc(X, test),
            'y_test': iloc(y, test)
        }
        for train, test in indexes
    ]


def predict_splits(
        clf: BaseEstimator,
        splits: List[Dict[str, PandasLike]],
        predict_proba: bool = False,
        predict_train: bool = False,
        clusters: Optional[List[dict]] = None,
        labels: Optional[List[str]] = None,
        fit_params: Optional[dict] = None,
        predict_params: Optional[dict] = None,
        n_jobs: int = -1,
        progress_bar: bool = False
) -> List[Dict[str, np.ndarray]]:
    """
    Aggregated splits prediction
    :param clf: predictor
    :param splits: aggregated splits
    :param predict_proba: if True returns probabilities of 1 instead of binary output
    :param predict_train: if True returns predictions for train set in addition to y_true, y_pred
    :param clusters:
    :param labels:
    :param fit_params: params for clf.fit
    :param predict_params: params for clf.predict
    :param n_jobs: count of jobs for joblib.Parallel
    :param progress_bar: whether to use tqdm (only for n_jobs=1)
    :return: (y_true, y_pred, y_train_pred) if predict_train else (y_true, y_pred)
    """

    fit_params = fit_params or {}
    predict_params = predict_params or {}

    def predict_split(clf, split, predict_proba, predict_train, fit_params, predict_params, clusters, labels):
        X_train, y_train, X_test, y_test = itemgetter('X_train', 'y_train', 'X_test', 'y_test')(split)

        fit_params = fit_params.copy()
        t1 = fit_params.pop('t1', None)
        clf.fit(
            X_train, y_train,
            **{
                **({'clusters': clusters} if 'clusters' in inspect.getfullargspec(clf.fit).args else {}),
                **({'labels': labels} if 'labels' in inspect.getfullargspec(clf.fit).args else {}),
                **({'t1': t1[X_train.index]} if t1 is not None else {}),
                **fit_params
            }
        )

        y_train_pred = None
        if predict_proba:
            y_pred = clf.predict_proba(X_test, **predict_params)[:, 1]
            if predict_train:
                y_train_pred = clf.predict_proba(X_train, **predict_params)[:, 1]
        else:
            y_pred = clf.predict(X_test, **predict_params)
            if predict_train:
                y_train_pred = clf.predict(X_train, **predict_params)

        res = {
            'y_true': np.array(y_test.values),
            'y_pred': y_pred,
            'index': X_test.index.values if hasattr(X_test, 'index') else None
        }

        if predict_train:
            res = {
                'y_train': np.array(y_train),
                'y_train_pred': y_train_pred,
                'train_index': X_train.index.values if hasattr(X_train, 'index') else X_train,
                **res
            }
        return res

    tqdm_ = lambda x: x
    if n_jobs == 1:
        parallel, fn = list, predict_split
        if progress_bar:
            tqdm_ = lambda x: tqdm(x, desc='splits processed')
    else:
        parallel, fn = Parallel(n_jobs=n_jobs), delayed(predict_split)

    return parallel(
        fn(clone(clf), split, predict_proba, predict_train, fit_params, predict_params, clusters, labels)
        for split in tqdm_(splits)
    )


def predicts_to_paths(
        predicts: List[Dict[str, np.ndarray]],
        k_tests: int,
        n_folds: int,
        test_folds_sizes: List[List[int]] = None
) -> List[Dict[str, np.ndarray]]:

    if k_tests > 1:
        n_paths = CombinatorialCV.get_n_paths(k_tests, n_folds)
    elif k_tests == 1:
        n_paths = 1
    else:
        raise ValueError('Unexpected value of k_tests')

    if k_tests == 1:
        return [utils.merge_dicts_array_values(predicts)]

    # place preds to folds
    preds_splits = [
        [None for _ in range(n_folds)]
        for _ in predicts
    ]
    if test_folds_sizes:
        def split_dict_array_values(predict, k_tests, split_i, test_folds_sizes):
            return utils.split_dict_array_values(predict, split_sizes=test_folds_sizes[split_i])

    else:
        def split_dict_array_values(predict, k_tests, split_i, test_folds_sizes):
            return utils.split_dict_array_values(predict, splits=k_tests)

    for split_i, (predict, test_ids) in enumerate(zip(predicts, combinations(range(n_folds), k_tests))):
        splitted_predicts = split_dict_array_values(predict, k_tests, split_i, test_folds_sizes)

        for fold_i, i in zip(test_ids, range(k_tests)):
            preds_splits[split_i][fold_i] = splitted_predicts[i]

    # calculate paths
    paths: List[List[Dict[str, np.ndarray]]] = []
    used_folds = set()

    for p in range(n_paths):
        paths.append([])
        last_fold_id = -1
        for split_id, folds in enumerate(combinations(range(n_folds), k_tests)):

            for fold_id in folds:
                if (split_id, fold_id) in used_folds:
                    continue

                if fold_id > last_fold_id:
                    last_fold_id = fold_id
                    used_folds.add((split_id, fold_id))

                    paths[-1].append(
                        preds_splits[split_id][fold_id]
                    )

                    if last_fold_id == n_folds - 1:  # full path found
                        break

    return [utils.merge_dicts_array_values(path_dicts) for path_dicts in paths]
