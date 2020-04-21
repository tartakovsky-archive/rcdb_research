from typing import Union, List, Tuple, Iterable, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold

PandasLike = Union[pd.DataFrame, pd.Series]
Split = Tuple[np.ndarray, np.ndarray]
Splits = List[Split]


class EmbargoedKFoldSplitterWithTainting(BaseCrossValidator):

    class SplitterException(Exception):
        pass

    def __init__(self, k_fold: int, embargo: int = None, tainted_up_to=None):
        if k_fold <= 1 and tainted_up_to is None:
            raise self.SplitterException(
                "No data left for training set. "
                "Either set k_fold to > 1 or mark some data as tainted by setting tainted_up_to to not None"
            )

        self.k_fold = k_fold
        self.embargo = embargo
        self.tainted_up_to = tainted_up_to

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.k_fold

    @staticmethod
    def split_by_index(data: PandasLike, index) -> Tuple[PandasLike, PandasLike]:
        return data[data.index <= index], data[data.index > index]

    @staticmethod
    def add_tainted_to_train(
        splits: Splits,
        tainted: np.ndarray
    ) -> Splits:
        return [
            (np.hstack((tainted, train)), test)
            for (train, test) in splits
        ]

    @staticmethod
    def kfold_splits(data, n_splits: int) -> Splits:
        return [
            (data[train_idx], data[test_idx])
            for train_idx, test_idx in KFold(n_splits=n_splits).split(data)
        ]

    @staticmethod
    def embargo_splits(splits: Splits, embargo: int) -> Splits:
        _splits = []
        for train, test in splits[:-1]:
            after_test = train > test[-1]
            before_test = ~after_test

            if np.sum(before_test):
                train = np.hstack((train[before_test], train[after_test][embargo:]))
            else:
                train = train[embargo:]

            _splits.append((train, test))

        _splits.append(splits[-1])
        return _splits

    def validate_splits(self, splits: Splits):
        for train, _ in splits:
            if not len(train):
                raise self.SplitterException(f'Not enough data left to perform {self.k_fold} folds')

    def split(self, X, *args, **kwargs) -> List[Split]:
        """
        Generate indices to split data into training and test set.

        :param array-like X: shape (n_samples, n_features).
                             Training data, where n_samples is the number of samples
                             and n_features is the number of features
        :param array-like y:
        :param array-like groups:
        :return: yield of train ndarray and test ndarray
        """
        if self.tainted_up_to is not None:
            if not isinstance(X, pd.DataFrame) or not isinstance(X, pd.Series):
                X = pd.DataFrame(X)

            X_tainted, X = self.split_by_index(X, self.tainted_up_to)
            tainted_idxs = np.arange(0, len(X_tainted))
            idxs = np.arange(len(X_tainted), len(X_tainted) + len(X))

            if not len(idxs) or (self.k_fold <= 1 and not len(tainted_idxs)):
                raise self.SplitterException('No data left for test set after separating tainted observations')

            if self.k_fold <= 1:
                return [(tainted_idxs, idxs)]

        else:
            tainted_idxs = np.array([])
            idxs = np.arange(len(X))

        try:
            splits = self.kfold_splits(idxs, self.k_fold)
        except ValueError:
            raise self.SplitterException(f'Not enough data left to perform {self.k_fold} folds')

        if self.embargo:
            splits = self.embargo_splits(splits, self.embargo)

        if self.tainted_up_to is not None:
            splits = self.add_tainted_to_train(splits, tainted_idxs)

        self.validate_splits(splits)

        return splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        return map(lambda split: split[1], self.split(X, y, groups))


def split_indexes_to_bars(
    X: Union[PandasLike, np.ndarray],
    y: Union[PandasLike, np.ndarray],
    indexes: Iterable[Split]
) -> List[Dict[str, PandasLike]]:

    if isinstance(X, np.ndarray):
        def iloc(data, idxs): return data[idxs]
    else:
        def iloc(data, idxs): return data.iloc[idxs]

    return [
        {
            'X_train': iloc(X, train),
            'y_train': iloc(y, train),
            'X_test': iloc(X, test),
            'y_test': iloc(y, test)
        }
        for train, test in indexes
    ]
