import numpy as np
from sklearn.base import is_classifier, clone
from sklearn.model_selection import BaseCrossValidator, check_cv
from sklearn.model_selection._validation import indexable, _num_samples, _fit_and_predict
from joblib import Parallel, delayed


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
        
        self.last_n_gap_size = None

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
        
        self.last_n_gap_size = n_gap
        
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


def cross_val_predict_timeseries_splits(estimator, X, y=None, groups=None, cv='warn',
                                        n_jobs=None, verbose=0, fit_params=None,
                                        pre_dispatch='2*n_jobs', method='predict', flatten=True):
    """
    Modification of:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
    https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/model_selection/_validation.py#L647

    For working with TimeSeriesSplit-based cross validation

    ---------
    Generate cross-validated estimates for each input data point
    The data is split according to the cv parameter. Each sample belongs
    to exactly one test set, and its prediction is computed with an
    estimator fitted on the corresponding training set.
    Passing these predictions into an evaluation metric may not be a valid
    way to measure generalization performance. Results can differ from
    `cross_validate` and `cross_val_score` unless all tests sets have equal
    size and the metric decomposes over samples.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.
    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.
    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" `cv` instance
        (e.g., `GroupKFold`).
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
        .. versionchanged:: 0.20
            ``cv`` default value if None will change from 3-fold to 5-fold
            in v0.22.
    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    verbose : integer, optional
        The verbosity level.
    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    method : string, optional, default: 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.
    Returns
    -------
    Tuple (y_true, y_pred) if flatten=True
    Array of by-split y_true, y_pred if flatten=False

    y_true : ndarray
    y_pred : ndarray
        This is the result of calling ``method``
    """
    def unpack(cvps_results):
        y_pred_all = []
        y_true_all = []
        for (y_pred, y_true) in cvps_results:
            y_pred_all += list(y_pred)
            y_true_all += list(y_true)
        return np.array(y_true_all), np.array(y_pred_all)

    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    result = [(pred_block_i, y[indxs]) for pred_block_i, indxs in prediction_blocks]

    return unpack(result) if flatten else result
