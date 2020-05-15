from operator import itemgetter
from itertools import combinations
from functools import reduce
from typing import Union, List, Tuple, Iterable, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from joblib import Parallel, delayed

from ... import utils

PandasLike = Union[pd.DataFrame, pd.Series]
Split = Tuple[np.ndarray, np.ndarray]
Splits = List[Split]


class CombinatorialCV(BaseCrossValidator):
    class SplitterException(Exception):
        pass

    def __init__(
            self,
            n_folds: int,
            embargo: Optional[int] = None,
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
        self.embargo = embargo
        self.tainted_up_to = tainted_up_to
        self.k_tests = k_tests

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
        if self.embargo and not all(len(g) - self.embargo > 0 for g in groups[self.k_tests:]):
            raise self.SplitterException('Not enough train set data to embargo')

        groups_idxs = np.arange(self.n_folds)

        splits = []

        for test_folds in combinations(range(self.n_folds), self.k_tests):
            train_folds = groups_idxs[~np.isin(groups_idxs, test_folds)]

            train_groups = [
                groups[i][self.embargo:] if self.embargo and any(i - ti == 1 for ti in test_folds) else groups[i]
                for i in train_folds
            ]
            train_groups.insert(0, tainted_idxs)
            train = np.hstack(train_groups)

            test = np.hstack((itemgetter(*test_folds)(groups)))

            splits.append((train, test))

        return splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        return map(lambda split: split[1], self.split(X, y, groups))


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
        fit_args: Optional[dict] = None,
        predict_args: Optional[dict] = None,
        n_jobs: int = -1
) -> List[Dict[str, np.ndarray]]:
    """
    Aggregated splits prediction
    :param clf: predictor
    :param splits: aggregated splits
    :param predict_proba: if True returns probabilities of 1 instead of binary output
    :param predict_train: if True returns predictions for train set in addition to y_true, y_pred
    :param fit_args: params for clf.fit
    :param predict_args: params for clf.predict
    :param n_jobs: count of jobs for joblib.Parallel
    :return: (y_true, y_pred, y_train_pred) if predict_train else (y_true, y_pred)
    """

    fit_args = fit_args or {}
    predict_args = predict_args or {}

    def predict_split(clf, split, predict_proba, predict_train, fit_args, predict_args):
        X_train, y_train, X_test, y_test = itemgetter('X_train', 'y_train', 'X_test', 'y_test')(split)

        clf.fit(X_train, y_train, **fit_args)

        y_train_pred = None
        if predict_proba:
            y_pred = clf.predict_proba(X_test, **predict_args)[:, 1]
            if predict_train:
                y_train_pred = clf.predict_proba(X_train, **predict_args)[:, 1]
        else:
            y_pred = clf.predict(X_test, **predict_args)
            if predict_train:
                y_train_pred = clf.predict(X_train, **predict_args)

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

    if n_jobs == 1:
        return [
            predict_split(clf, split, predict_proba, predict_train, fit_args, predict_args)
            for split in splits
        ]

    parallel = Parallel(n_jobs=n_jobs)
    return parallel(
        delayed(predict_split)(clone(clf), split, predict_proba,
                               predict_train, fit_args, predict_args)
        for split in splits
    )


def predicts_to_paths(predicts: List[Dict[str, np.ndarray]], k_tests: int, n_folds: int) -> List[Dict[str, np.ndarray]]:
    if k_tests > 1:
        n_paths = CombinatorialCV.get_n_paths(k_tests, n_folds)
    elif k_tests == 1:
        n_paths = 1
    else:
        raise ValueError('Unexpected value of k_tests')

    # place preds to folds
    preds_splits = [
        [None for _ in range(n_folds)]
        for _ in predicts
    ]
    for split_i, (predict, test_ids) in enumerate(zip(predicts, combinations(range(n_folds), k_tests))):
        splitted_predicts = utils.split_dict_array_values(predict, k_tests)
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
