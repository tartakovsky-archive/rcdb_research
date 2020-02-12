from typing import Union, Callable
import weakref

import numpy as np
import pandas as pd

from ...numpy_ext import rolling_apply
from ..entities import Predictions


class PredictionMetrics:
    """
    Class for analyzing Predictions objects outputted by ML models
    """

    ############
    # Initialization
    ############

    def __init__(self, preds: Predictions, direction: str = 'pos', labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}):
        """
        :param preds: instance of Predictions class
        :param direction:
        :param labels:

        """
        self.preds = weakref.proxy(preds)
        self.direction = direction
        self.labels = labels

    def score(
            self,
            score_fn: Callable,
            window: int = None,
            dense: bool = False,
            raw: bool = False
    ) -> Union[pd.Series, tuple, float]:
        """
        Calculates score function
        :param score_fn: score function
        :param window: rolling window size, if is not None, score calculates on rolling window
        :param dense: drop zeroes from the array when True
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:    
        """

        index = self.preds.index
        y_true = self.preds.y_true
        y_pred = self.preds.y_pred

        if dense:
            if self.direction == 'both':
                mask = y_pred != self.labels['neu']
            else:
                mask = y_pred == self.labels[self.direction]

            index = index[mask]
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        if window is None:
            sc = score_fn(y_true, y_pred, self.direction, self.labels)
        else:
            sc = rolling_apply(score_fn, window, y_true, y_pred, direction=self.direction, labels=self.labels)

        if type(sc) is np.ndarray:
            return (sc, index) if raw else pd.Series(sc, index=index)
        else:
            return sc

    ############
    # Public methods
    ############

    def accuracy(self, window: int = None, raw: bool = False) -> Union[pd.Series, tuple, float]:
        """
        accuracy
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.accuracy_score, window=window, raw=raw)

    def precision(self, window=None, dense=False, raw=False) -> Union[pd.Series, tuple, float]:
        """
        precision
        :param window: rolling window size, if window is None then returns a scalar
        :param dense: drop zeroes from the array when True
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.precision_score, window=window, dense=dense, raw=raw)

    def recall(self, window=None, raw=False) -> Union[pd.Series, tuple, float]:
        """
        recall
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.recall_score, window=window, raw=raw)

    def activity(self, window=None, raw=False) -> Union[pd.Series, tuple, float]:
        """
        activity = positives / observations, percentage of the bars with non-labels['neu'] predictions
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.activity_score, window=window, raw=raw)

    def positives(self, raw=False) -> Union[pd.Series, tuple]:
        """
        positives
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.positives_score, raw=raw)

    def negatives(self, raw=False) -> Union[pd.Series, tuple]:
        """
        negatives
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.negatives_score, raw=raw)

    def n_positives(self, window=None, raw=False) -> Union[pd.Series, tuple, float]:
        """
        n_positives
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.n_positives_score, window=window, raw=raw)

    def n_negatives(self, window=None, raw=False) -> Union[pd.Series, tuple, float]:
        """
        n_negatives
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.n_negatives_score, window=window, raw=raw)

    def tp(self, raw=False) -> Union[pd.Series, tuple]:
        """
        tp
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.tp_score, raw=raw)

    def fp(self, raw=False) -> Union[pd.Series, tuple]:
        """
        fp
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.fp_score, raw=raw)

    def tn(self, raw=False) -> Union[pd.Series, tuple]:
        """
        tn
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.tn_score, raw=raw)

    def fn(self, raw=False) -> Union[pd.Series, tuple]:
        """
        fn
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.fn_score, raw=raw)

    def n_tp(self, window=None, raw=False) -> Union[pd.Series, tuple, float]:
        """
        Number of true positives
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.n_tp_score, window=window, raw=raw)

    def n_fp(self, window=None, raw=False) -> Union[pd.Series, tuple, float]:
        """
        Number of false positives
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.n_fp_score, window=window, raw=raw)

    def n_tn(self, window=None, raw=False) -> Union[pd.Series, tuple, float]:
        """
        Number of true negatives
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.n_tn_score, window=window, raw=raw)

    def n_fn(self, window=None, raw=False) -> Union[pd.Series, tuple, float]:
        """
        Number of false negatives
        :param window: rolling window size, if window is None then returns a scalar
        :param raw: if True returns a tuple of np.ndarray and index, otherwise pd.Series
        :return:
        """
        return self.score(Scores.n_fn_score, window=window, raw=raw)

    def dataframe(self) -> pd.DataFrame:
        metrics_dict = dict(
            precision=self.precision(),
            activity=self.activity(),
            tp=self.n_tp(),
            fp=self.n_fp(),
            tn=self.n_tn(),
            fn=self.n_fn(),
            positives=self.n_positives(),
            negatives=self.n_negatives(),
            observations=self.tp().size,
        )
        metrics_df = pd.DataFrame({**metrics_dict}, index=[0])
        return metrics_df


############
# Scoring
############

class Scores:
    @staticmethod
    def tp_score(y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> np.ndarray:

        if direction == 'both':
            return ((y_pred == y_true) & (y_pred != labels['neu'])).astype(np.int8)
        else:
            return ((y_pred == labels[direction]) & (y_true == labels[direction])).astype(np.int8)

    @staticmethod
    def fp_score(y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> np.ndarray:

        if direction == 'both':
            return ((y_pred != y_true) & (y_pred != labels['neu'])).astype(np.int8)
        else:
            return ((y_pred == labels[direction]) & (y_true != labels[direction])).astype(np.int8)

    @staticmethod
    def tn_score(y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> np.ndarray:
        if direction == 'both':
            return ((y_pred == labels['neu']) & (y_true == labels['neu'])).astype(np.int8)
        else:
            return ((y_pred != labels[direction]) & (y_true != labels[direction])).astype(np.int8)

    @staticmethod
    def fn_score(y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> np.ndarray:

        if direction == 'both':
            return ((y_pred == labels['neu']) & (y_true != labels['neu'])).astype(np.int8)
        else:
            return ((y_pred != labels[direction]) & (y_true == labels[direction])).astype(np.int8)

    @classmethod
    def n_tp_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> int:

        return cls.tp_score(y_true, y_pred, direction, labels).sum()

    @classmethod
    def n_fp_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> int:

        return cls.fp_score(y_true, y_pred, direction, labels).sum()

    @classmethod
    def n_tn_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> int:
        return cls.tn_score(y_true, y_pred, direction, labels).sum()

    @classmethod
    def n_fn_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> int:
        return cls.fn_score(y_true, y_pred, direction, labels).sum()

    @classmethod
    def positives_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> np.ndarray:
        tp = cls.tp_score(y_true, y_pred, direction, labels)
        fp = cls.fp_score(y_true, y_pred, direction, labels)
        return tp - fp

    @classmethod
    def negatives_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> np.ndarray:

        tn = cls.tn_score(y_true, y_pred, direction, labels)
        fn = cls.fn_score(y_true, y_pred, direction, labels)
        return tn - fn

    @classmethod
    def n_positives_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> int:

        return np.count_nonzero(cls.positives_score(y_true, y_pred, direction, labels))

    @classmethod
    def n_negatives_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> int:

        return np.count_nonzero(cls.negatives_score(y_true, y_pred, direction, labels))

    @classmethod
    def accuracy_score(cls, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

        return (y_true == y_pred).sum() / y_true.size

    @classmethod
    def precision_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> float:

        tp = cls.n_tp_score(y_true, y_pred, direction, labels)
        fp = cls.n_fp_score(y_true, y_pred, direction, labels)
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    @classmethod
    def recall_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> float:

        tp = cls.n_tp_score(y_true, y_pred, direction, labels)
        fn = cls.n_fn_score(y_true, y_pred, direction, labels)
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    @classmethod
    def activity_score(cls, y_true: np.ndarray, y_pred: np.ndarray, direction: str, labels: dict) -> float:

        return cls.n_positives_score(y_true, y_pred, direction, labels) / y_true.size
