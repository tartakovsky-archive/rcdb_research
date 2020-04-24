from operator import itemgetter
from itertools import combinations
from functools import reduce
from typing import Union, List, Tuple, Iterable, Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from joblib import Parallel, delayed

from .. import utils

PandasLike = Union[pd.DataFrame, pd.Series]
Split = Tuple[np.ndarray, np.ndarray]
Splits = List[Split]


class CombinatorialKFold(BaseCrossValidator):
    class SplitterException(Exception):
        pass

    def __init__(
        self,
        k_fold: int,
        embargo: Optional[int] = None,
        tainted_up_to: Optional[Any] = None,
        n_test: int = 1
    ):
        if k_fold <= 1 and tainted_up_to is None:
            raise self.SplitterException(
                "No data left for training set. "
                "Either set k_fold to > 1 or mark some data as tainted by setting tainted_up_to to not None"
            )

        if k_fold <= n_test and not (k_fold <= 1 and n_test == 1):
            raise ValueError('k_fold value must be higher then n_test')

        self.k_fold = k_fold
        self.embargo = embargo
        self.tainted_up_to = tainted_up_to
        self.n_test = n_test

    @staticmethod
    def get_n_paths(n_test, k_fold):
        return reduce(lambda a, b: a * b, map(lambda i: k_fold - i, range(1, n_test))) // np.math.factorial(n_test - 1)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return reduce(
            lambda a, b: a * b,
            map(lambda i: self.k_fold - i, range(self.n_test))
        ) // np.math.factorial(self.n_test)

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

            if not len(idxs) or (self.k_fold <= 1 and not len(tainted_idxs)):
                raise self.SplitterException('No data left for test set after separating tainted observations')
        else:
            tainted_idxs = np.array([], dtype=np.int)
            idxs = np.arange(len(X))

        return tainted_idxs, idxs

    def split(self, X: Union[PandasLike, np.ndarray], *args, **kwargs) -> List[Split]:
        tainted_idxs, idxs = self.split_by_tainted(X)

        if self.k_fold <= 1 and self.n_test == 1:
            return [(tainted_idxs, idxs)]

        if len(idxs) < self.k_fold:
            raise self.SplitterException(f'Not enough data left to perform {self.k_fold} folds')

        groups = np.array_split(idxs, self.k_fold)
        if self.embargo and not all(len(g) - self.embargo > 0 for g in groups[self.n_test:]):
            raise self.SplitterException('Not enough train set data to embargo')

        groups_idxs = np.arange(self.k_fold)

        splits = []

        for test_folds in combinations(range(self.k_fold), self.n_test):
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
) -> List[Dict[str, PandasLike]]:

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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregated splits prediction
    :param cls: predictor
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

    parallel = Parallel(n_jobs=n_jobs)
    return parallel(
        delayed(predict_split)(clone(clf), split, predict_proba,
                               predict_train, fit_args, predict_args)
        for split in splits
    )


def predicts_to_paths(predicts: List[Dict[str, np.ndarray]], n_test: int, k_fold: int) -> List[Dict[str, np.ndarray]]:
    if n_test > 1:
        n_paths = CombinatorialKFold.get_n_paths(n_test, k_fold)
    elif n_test == 1:
        n_paths = 1
    else:
        raise ValueError('Unexpected value of n_test')

    # place preds to folds
    preds_splits = [
        [None for _ in range(k_fold)]
        for _ in predicts
    ]
    for split_i, (predict, test_ids) in enumerate(zip(predicts, combinations(range(k_fold), n_test))):
        splitted_predicts = utils.split_dict_array_values(predict, n_test)
        for fold_i, i in zip(test_ids, range(n_test)):
            preds_splits[split_i][fold_i] = splitted_predicts[i]

    # calculate paths
    paths: List[List[Dict[str, np.ndarray]]] = []
    used_folds = set()

    for p in range(n_paths):
        paths.append([])
        last_fold_id = -1
        for split_id, folds in enumerate(combinations(range(k_fold), n_test)):

            for fold_id in folds:
                if (split_id, fold_id) in used_folds:
                    continue

                if fold_id > last_fold_id:
                    last_fold_id = fold_id
                    used_folds.add((split_id, fold_id))

                    paths[-1].append(
                        preds_splits[split_id][fold_id]
                    )

                    if last_fold_id == k_fold - 1:  # full path found
                        print(paths[-1])
                        break

    return [utils.merge_dicts_array_values(path_dicts) for path_dicts in paths]
