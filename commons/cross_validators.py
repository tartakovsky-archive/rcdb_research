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
    def __init__(self, n_splits, test_size=None, train_size=None, gap_size=0, expanding=False, is_fixed=False):

        if not (train_size or test_size):
            raise ValueError("Please provide test_size or/and train_size parameter")

        if is_fixed and train_size and test_size:
            self.all_fixed_sizes_set = True
            test_size = int(test_size)
            train_size = int(train_size)
            gap_size = int(gap_size)
        else:
            self.all_fixed_sizes_set = False

        if not is_fixed:
            if train_size and test_size:
                raise ValueError("Please provide only test_size or train_size parameter")

            if train_size is not None:
                check_size = train_size
                msg_part = "test"
            else:
                check_size = test_size
                msg_part = "train"

            if not (check_size > 0 and 0 < gap_size + check_size < 1):
                raise ValueError(f"Not enough {msg_part} part")

        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.train_size = train_size
        self.gap_size = gap_size
        self.expanding = expanding
        self.is_fixed = is_fixed

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

    def _calculate_sizes_fixed(self, n_samples):
        """
        Calculate split parts if sizes is bars count
        :param n_samples: input data size
        :return: tuple of n_wf_split, n_train, n_gap, n_test
        """
        n_test = self.test_size or int((n_samples - self.train_size) / self.n_splits)
        n_wf_split = n_samples - (self.n_splits - 1) * n_test
        n_train = n_wf_split - n_test
        return n_wf_split, n_train, self.gap_size, n_test

    def _calculate_sizes(self, n_samples):
        """
        Calculate split parts if sizes is percents
        :param n_samples: input data size
        :return: tuple of n_wf_split, n_train, n_gap, n_test
        """
        if self.n_splits == 1:
            n_wf_split = n_samples
        else:
            test_size_percents = self.test_size or (1.0 - self.train_size - self.gap_size)

            n_wf_split = self.get_n_wf_split(
                n_samples,
                test_size_percents,
                self.n_splits
            )

        if self.test_size:
            raw_n_test = n_wf_split * self.test_size
            raw_n_train = n_wf_split - raw_n_test
        else:
            raw_n_train = n_wf_split * self.train_size
            raw_n_test = n_wf_split - raw_n_train

        raw_n_gap = n_wf_split * self.gap_size

        return int(n_wf_split), int(raw_n_train), int(raw_n_gap), int(raw_n_test)

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

        if self.all_fixed_sizes_set \
                and self.train_size + self.gap_size + self.n_splits * self.test_size > n_samples:
            raise ValueError("Provide more data")

        indices = np.arange(n_samples)

        (n_wf_split, n_train, n_gap, n_test)\
            = self._calculate_sizes_fixed(n_samples) if self.is_fixed else self._calculate_sizes(n_samples)

        if n_test <= 0 or n_train <= 0:
            raise ValueError(
                f"Couldn't build splits(n_train={n_train}, n_test={n_test}). "
                f"Set less n_splits or provide more data"
            )

        if self.test_size:
            n_train -= n_gap
        else:
            n_test -= n_gap

        rest = n_samples - int(n_wf_split) - (n_test * (self.n_splits - 1))
        # print("n_wf_split", n_wf_split, "n_train", n_train, "n_gap", n_gap, "n_test", n_test, "rest", rest)

        train_start = 0
        rest_pieces = (rest // self.n_splits) or 1
        for split_number in range(1, self.n_splits + 1):
            train_end = train_start + n_train
            test_start = train_end + n_gap

            additional_train_start = 0
            if split_number == self.n_splits:
                test_end = n_samples
            else:
                test_end = test_start + n_test
                if rest:
                    test_end += rest_pieces
                    rest -= rest_pieces
                    additional_train_start = rest_pieces

            train_set_slice_start = 0 if self.expanding else train_start

            if self.all_fixed_sizes_set and not self.expanding:
                train_set_slice_start = train_end - self.train_size
                if train_set_slice_start < 0:
                    train_set_slice_start = 0

            train_split = indices[train_set_slice_start:train_end]
            test_split = indices[test_start:test_end]

            yield train_split, test_split

            train_start = train_start + n_test + additional_train_start
