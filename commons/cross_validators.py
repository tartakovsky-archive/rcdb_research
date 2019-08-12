import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._validation import indexable, _num_samples


class WalkForwardCV(BaseCrossValidator):
    """
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets with unique test set for each split.
    Cross-validator support gap between test and train set(by default it size is 0).

    Example of 3-split time series cross-validation on a dataset with 10 samples::

    >>> import numpy as np
    >>> from sklearn.model_selection import WalkForward
    >>> X = np.arange(10)
    >>> y = np.arange(10)
    >>> wf = WalkForwardCV(n_splits=3, test_size=.25, gap_size=.2)
    >>> print(wf)
    WalkForwardCV(expanding=False, gap_size=0.2, n_splits=3, test_size=0.25)
    >>> for train_index, test_index in wf.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2] TEST: [4 5]
    TRAIN: [2 3 4] TEST: [6 7]
    TRAIN: [4 5 6] TEST: [8 9]
    >>> wfe = WalkForwardCV(n_splits=3, test_size=.25, gap_size=.2, expanding=True)
    >>> print(wfe)
    WalkForwardCV(expanding=True, gap_size=0.2, n_splits=3, test_size=0.25)
    >>> for train_index, test_index in wfe.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 1 2] TEST: [4 5]
    TRAIN: [0 1 2 3 4] TEST: [6 7]
    TRAIN: [0 1 2 3 4 5 6] TEST: [8 9]
    """
    def __init__(self, n_splits, test_size, gap_size=.0, expanding=False):

        if not (test_size > 0 and 0 < gap_size + test_size < 1):
            raise ValueError("Not enough train part")

        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.gap_size = gap_size
        self.expanding = expanding

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator

        :param X: Always ignored, exists for compatibility
        :param y: Always ignored, exists for compatibility
        :param groups: Always ignored, exists for compatibility
        :return: the number of splitting iterations in the cross-validator
        """
        return self.n_splits

    @staticmethod
    def get_n_wf_split(n_samples, offset_pct, n_splits):
        """
        Calculate observable data in split

        :param int n_samples: data size
        :param float offset_pct: the percentage of non-overlapping data in each split
        :param int n_splits: number of splits
        :return: size of observable data in split
        :rtype: int
        """
        a = offset_pct * (n_splits - 1)
        return (n_samples / a) / (1.0 + 1.0 / a)

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        :param array-like X: shape (n_samples, n_features).
                             Training data, where n_samples is the number of samples
                             and n_features is the number of features
        :param array-like y:
        :param array-like groups:
        :return: yield of train ndarray and test ndarray
        """

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits "
                f"n_splits={self.n_splits} greater"
                f" than the number of samples: n_samples={n_samples}."
            )

        indices = np.arange(n_samples)

        if self.n_splits == 1:
            n_wf_split = n_samples

        else:
            n_wf_split = self.get_n_wf_split(
                n_samples, self.test_size, self.n_splits
            )

        raw_n_test = n_wf_split * self.test_size
        raw_n_gap = n_wf_split * self.gap_size
        raw_n_train = n_wf_split - raw_n_test - raw_n_gap

        n_test = int(raw_n_test)
        n_gap = int(raw_n_gap)
        n_train = int(raw_n_train)

        if not n_test or not n_train:
            raise ValueError(
                f"Couldn't build splits(n_train={n_train}, n_test={n_test}). "
                f"Set less n_splits or provide much data"
            )

        rest = n_samples - int(n_wf_split) - (n_test * (self.n_splits - 1))

        train_start = 0
        for split_number in range(1, self.n_splits + 1):
            train_end = train_start + n_train
            test_start = train_end + n_gap

            additional_train_start = 0
            if split_number == self.n_splits:
                test_end = n_samples
            else:
                test_end = test_start + n_test
                if rest:
                    test_end += 1
                    rest -= 1
                    additional_train_start = 1

            train_set_slice_start = 0 if self.expanding else train_start
            train_split = indices[train_set_slice_start:train_end]
            test_split = indices[test_start:test_end]

            yield train_split, test_split

            train_start = train_start + n_test + additional_train_start
