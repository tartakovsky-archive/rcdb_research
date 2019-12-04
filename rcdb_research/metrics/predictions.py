import logging
from typing import Union, Callable

import numpy as np
import pandas as pd

from ..numpy_ext import rolling_apply


class Predictions:
    """
    Class for analyzing predictions of ML models
    """
    ############
    # Initialization
    ############

    def __init__(self, y_true: np.array, y_pred: np.array, index: np.array = None):

        if index is not None:
            index = index.copy()

            if y_true.size > index.size:
                raise ValueError(f'index.size={index.size} should be >= y_true.size={y_true.size}')

            if y_true.size != index.size:
                logging.warning(
                    f' Last {y_true.size} out of {index.size} '
                    'elements will be taken from index to match y_true size'
                )

            index = index[-y_true.size:]

        if y_pred.size != y_true.size:
            raise ValueError(f"Size of y_pred should be same size with y_true")

        self.y_true = y_true
        self.y_pred = y_pred
        self.index = index

    ############
    # Scoring
    ############
    class Scores:
        @staticmethod
        def tp_score(y_true: np.array, y_pred: np.array,
                     label: Union[str, int], neg_label: int) -> np.array:

            if label == 'all':
                return ((y_pred == y_true) & (y_pred != neg_label)).astype(np.int8)
            else:
                return ((y_pred == label) & (y_true == label)).astype(np.int8)

        @staticmethod
        def fp_score(y_true: np.array, y_pred: np.array,
                     label: Union[str, int], neg_label: int) -> np.array:

            if label == 'all':
                return ((y_pred != y_true) & (y_pred != neg_label)).astype(np.int8)
            else:
                return ((y_pred == label) & (y_true != label)).astype(np.int8)

        @staticmethod
        def tn_score(y_true: np.array, y_pred: np.array,
                     label: Union[str, int], neg_label: int) -> np.array:
            if label == 'all':
                return ((y_pred == neg_label) & (y_true == neg_label)).astype(np.int8)
            else:
                return ((y_pred != label) & (y_true != label)).astype(np.int8)

        @staticmethod
        def fn_score(y_true: np.array, y_pred: np.array,
                     label: Union[str, int], neg_label: int) -> np.array:
            if label == 'all':
                return ((y_pred == neg_label) & (y_true != neg_label)).astype(np.int8)
            else:
                return ((y_pred != label) & (y_true == label)).astype(np.int8)

        @classmethod
        def n_tp_score(cls, y_true: np.array, y_pred: np.array,
                       label: Union[str, int], neg_label: int) -> int:

            return cls.tp_score(y_true, y_pred, label, neg_label).sum()

        @classmethod
        def n_fp_score(cls, y_true: np.array, y_pred: np.array,
                       label: Union[str, int], neg_label: int) -> int:

            return cls.fp_score(y_true, y_pred, label, neg_label).sum()

        @classmethod
        def n_tn_score(cls, y_true: np.array, y_pred: np.array,
                       label: Union[str, int], neg_label: int) -> int:
            return cls.tn_score(y_true, y_pred, label, neg_label).sum()

        @classmethod
        def n_fn_score(cls, y_true: np.array, y_pred: np.array,
                       label: Union[str, int], neg_label: int) -> int:
            return cls.fn_score(y_true, y_pred, label, neg_label).sum()

        @classmethod
        def positives_score(cls, y_true: np.array, y_pred: np.array,
                            label: Union[str, int], neg_label: int) -> np.array:
            return cls.tp_score(y_true, y_pred, label, neg_label) - cls.fp_score(y_true, y_pred, label, neg_label)

        @classmethod
        def negatives_score(cls, y_true: np.array, y_pred: np.array,
                            label: Union[str, int], neg_label: int) -> np.array:

            return cls.tn_score(y_true, y_pred, label, neg_label) - cls.fn_score(y_true, y_pred, label, neg_label)

        @classmethod
        def n_positives_score(cls, y_true: np.array, y_pred: np.array,
                              label: Union[str, int], neg_label: int) -> int:

            return np.count_nonzero(cls.positives_score(y_true, y_pred, label, neg_label))

        @classmethod
        def n_negatives_score(cls, y_true: np.array, y_pred: np.array,
                              label: Union[str, int], neg_label: int) -> int:

            return np.count_nonzero(cls.negatives_score(y_true, y_pred, label, neg_label))

        @classmethod
        def accuracy_score(cls, y_true: np.array, y_pred: np.array,
                           label: Union[str, int], neg_label: int) -> np.array:

            return (y_true == y_pred).sum() / y_true.size

        @classmethod
        def precision_score(cls, y_true: np.array, y_pred: np.array,
                            label: Union[str, int], neg_label: int) -> float:

            tp = cls.n_tp_score(y_true, y_pred, label, neg_label)
            fp = cls.n_fp_score(y_true, y_pred, label, neg_label)
            return tp / (tp + fp) if (tp + fp) > 0 else 0

        @classmethod
        def recall_score(cls, y_true: np.array, y_pred: np.array,
                         label: Union[str, int], neg_label: int) -> float:

            tp = cls.n_tp_score(y_true, y_pred, label, neg_label)
            fn = cls.n_fn_score(y_true, y_pred, label, neg_label)
            return tp / (tp + fn) if (tp + fn) > 0 else 0

        @classmethod
        def activity_score(cls, y_true: np.array, y_pred: np.array,
                           label: Union[str, int], neg_label: int) -> float:

            return cls.n_positives_score(y_true, y_pred, label, neg_label) / y_true.size

    def score(
        self,
        score_fn: Callable,
        window: int = None,
        label: Union[str, int] = 'all',
        neg_label: int = 0,
        dense: bool = False,
        raw: bool = False
    ) -> Union[pd.Series, tuple, float]:
        """
        Calculates score function
        :param score_fn: score function
        :param window: rolling window size, if is not None, score calculates on rolling window
        :param label: 'all' or value of the label to calculate the score for, e.g. 1 or -1
        :param neg_label: value of the label for the negative class, usually 0
        :param dense: drop zeroes from the array when True
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """

        index = self.index
        y_true = self.y_true
        y_pred = self.y_pred

        if dense:
            ids = (y_pred != neg_label) if label == 'all' else (y_pred == label)
            index = index[ids]
            y_true = y_true[ids]
            y_pred = y_pred[ids]

        if window is None:
            sc = score_fn(y_true, y_pred, label, neg_label)
        else:
            sc = rolling_apply(score_fn, window, y_true, y_pred, label=label, neg_label=neg_label)

        if type(sc) is np.ndarray:
            return (sc, index) if raw else pd.Series(sc, index=index)
        else:
            return sc

    ############
    # Public methods
    ############

    def head(self, n: int) -> 'Predictions':
        """
        Returns copy of Predictions with first n items
        :param n: number of first items
        :return:
        """
        return Predictions(
            y_true=self.y_true[:n],
            y_pred=self.y_pred[:n],
            index=self.index[:n]
        )

    def tail(self, n: int) -> 'Predictions':
        """
        Returns copy of Predictions with last n items
        :param n: number of last items
        :return:
        """
        return Predictions(
            y_true=self.y_true[-n:],
            y_pred=self.y_pred[-n:],
            index=self.index[-n:]
        )

    def accuracy(self, window: int = None,
                 label='all', neg_label=0, raw: bool = False) -> Union[pd.Series, tuple, float]:
        """
        accuracy
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.accuracy_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def precision(self, window=None,
                  label='all', neg_label=0, dense=False, raw=False) -> Union[pd.Series, tuple, float]:
        """
        precision
        :param window: rolling window size, if window is None then returns scalar
        :param dense: drop zeroes from the array when True
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.precision_score, window=window,
                          label=label, neg_label=neg_label,
                          dense=dense, raw=raw)

    def recall(self, window=None, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple, float]:
        """
        recall
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.recall_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def activity(self, window=None, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple, float]:
        """
        activity = positives / observations, percentage of the bars with non-neg_label predictions
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.activity_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def positives(self, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple]:
        """
        positives
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.positives_score, label=label, neg_label=neg_label, raw=raw)

    def negatives(self, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple]:
        """
        negatives
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.negatives_score, label=label, neg_label=neg_label, raw=raw)

    def n_positives(self, window=None, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple, float]:
        """
        n_positives
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.n_positives_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def n_negatives(self, window=None, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple, float]:
        """
        n_negatives
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.n_negatives_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def tp(self, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple]:
        """
        tp
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.tp_score, label=label, neg_label=neg_label, raw=raw)

    def fp(self, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple]:
        """
        fp
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.fp_score, label=label, neg_label=neg_label, raw=raw)

    def tn(self, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple]:
        """
        tn
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.tn_score, label=label, neg_label=neg_label, raw=raw)

    def fn(self, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple]:
        """
        fn
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.fn_score, label=label, neg_label=neg_label, raw=raw)

    def n_tp(self, window=None, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple, float]:
        """
        Number of true positives
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.n_tp_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def n_fp(self, window=None, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple, float]:
        """
        Number of false positives
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.n_fp_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def n_tn(self, window=None, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple, float]:
        """
        Number of true negatives
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.n_tn_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def n_fn(self, window=None, label='all', neg_label=0, raw=False) -> Union[pd.Series, tuple, float]:
        """
        Number of false negatives
        :param window: rolling window size, if window is None then returns scalar
        :param raw: if True returns tuple of np.array and index, otherwise pd.Series
        :return:
        """
        return self.score(self.Scores.n_fn_score, window=window, label=label, neg_label=neg_label, raw=raw)

    def metrics(self, label='all', neg_label=0) -> pd.DataFrame:
        metrics_dict = dict(
            precision=self.precision(label=label, neg_label=neg_label),
            activity=self.activity(label=label, neg_label=neg_label),
            tp=self.n_tp(label=label, neg_label=neg_label),
            fp=self.n_fp(label=label, neg_label=neg_label),
            tn=self.n_tn(label=label, neg_label=neg_label),
            fn=self.n_fn(label=label, neg_label=neg_label),
            positives=self.n_positives(label=label, neg_label=neg_label),
            negatives=self.n_negatives(label=label, neg_label=neg_label),
            observations=self.y_pred.size,
        )
        metrics_df = pd.DataFrame({**metrics_dict}, index=[0])
        return metrics_df
