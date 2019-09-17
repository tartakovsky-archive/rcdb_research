import numpy as np
import pandas as pd
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

def cross_val_predict_splits(estimator, X, y=None, groups=None, cv='warn',
                             n_jobs=None, verbose=0, fit_params=None,
                             pre_dispatch='2*n_jobs', method='predict'):
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
    predictions : ndarray
        This is the result of calling ``method``
    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_predict
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> y_pred = cross_val_predict(lasso, X, y, cv=3)
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train, test, verbose, fit_params, method)
        for train, test in cv.split(X, y, groups))

    return [(pred_block_i, y[indxs]) for pred_block_i, indxs in prediction_blocks]


class CVResult:
    """
    Class for working with cv results gathered from cross_val_predict_splits(...)
    """
    ############
    # Initialization
    ############
    def __init__(self, y_pred, y_true):
        self.data = pd.DataFrame(dict(y_pred=y_pred, y_true=y_true))
        self.data.index = y_true

    @classmethod
    def from_cross_val_predict_results(cls, cvp_results):
        return cls(*cls.__unpack_predictions(cvp_results))

    @staticmethod
    def __unpack_predictions(cvp_results):
        y_pred_all = []
        y_true_all = []
        for (y_pred, y_true) in cvp_results:
            y_pred_all += list(y_pred)
            y_true_all += list(y_true)
        return np.array(y_pred_all), np.array(y_true_all)

    ############
    # Scoring
    ############

    class scores:
        @staticmethod
        def tp(y_true, y_pred):
            return np.where((y_pred == 1) & (y_true == 1), 1, 0)

        @staticmethod
        def fp(y_true, y_pred):
            return np.where((y_pred == 1) & (y_true == 0), 1, 0)

        @staticmethod
        def tn(y_true, y_pred):
            return np.where((y_pred == 0) & (y_true == 0), 1, 0)

        @staticmethod
        def fn(y_true, y_pred):
            return np.where((y_pred == 0) & (y_true == 1), 1, 0)

        @classmethod
        def accuracy_score(cls, y_true, y_pred):
            return np.where(y_true == y_pred, 1, 0).sum() / y_true.size

        @classmethod
        def precision_score(cls, y_true, y_pred):
            tp = cls.tp(y_true, y_pred).sum()
            fp = cls.fp(y_true, y_pred).sum()
            return tp / (tp + fp) if (tp + fp) > 0 else 0

        @classmethod
        def recall_score(cls, y_true, y_pred):
            tp = cls.tp(y_true, y_pred).sum()
            fn = cls.fn(y_true, y_pred).sum()
            return tp / (tp + fn) if (tp + fn) > 0 else 0

    @staticmethod
    def get_score_wrapper(score_fn):
        def score_wrapper(data):
            return score_fn(data.index.values, data.values)
        return score_wrapper

    def score(self, score_fn, window=None, sparse=True):
        score_wrapper = self.get_score_wrapper(score_fn)
        v = self.data.y_pred
        if not sparse:
            v = self.data.y_pred[self.data.y_pred.values != 0]

        if window is None:
            return score_wrapper(v)

        return v.rolling(window).apply(score_wrapper, raw=False).values

    ############
    # Public methods
    ############

    def accuracy(self, window=None, sparse=True):
        return self.score(self.scores.accuracy_score, window=window, sparse=sparse)

    def precision(self, window=None, sparse=True):
        return self.score(self.scores.precision_score, window=window, sparse=sparse)

    def recall(self, window=None, sparse=True):
        return self.score(self.scores.recall_score, window=window, sparse=sparse)

    def positives(self, sparse=True):
        return self.tp(sparse=sparse) - self.fp(sparse=sparse)

    def negatives(self, sparse=True):
        return self.tn(sparse=sparse) - self.fn(sparse=sparse)

    def tp(self, sparse=True):
        return self.score(self.scores.tp, window=None, sparse=sparse)

    def fp(self, sparse=True):
        return self.score(self.scores.fp, window=None, sparse=sparse)

    def tn(self, sparse=True):
        return self.score(self.scores.tn, window=None, sparse=sparse)

    def fn(self, sparse=True):
        return self.score(self.scores.fn, window=None, sparse=sparse)

    @property
    def y_pred(self):
        return self.data.y_pred.values

    @property
    def y_true(self):
        return self.data.y_true.values
