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

    def __init__(self, preds: 'Predictions', target_label: Union[str, int] = 'all', neu_label: int = 0):
        """
        :param preds: instance of Predictions class
        :param target_label: 'all' or value of the target_label to calculate the score for, e.g. 1 or -1
        :param neu_label: value of the target_label for the negative class, usually 0
        """
        self.preds = weakref.proxy(preds)
        self.target_label = target_label
        self.neu_label = neu_label

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
            ids = (y_pred != self.neu_label) if self.target_label == 'all' else (y_pred == self.target_label)
            index = index[ids]
            y_true = y_true[ids]
            y_pred = y_pred[ids]

        if window is None:
            sc = score_fn(y_true, y_pred, self.target_label, self.neu_label)
        else:
            sc = rolling_apply(score_fn, window, y_true, y_pred,
                               target_label=self.target_label, neu_label=self.neu_label)

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
        activity = positives / observations, percentage of the bars with non-neu_label predictions
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
            observations=self.y_pred.size,
        )
        metrics_df = pd.DataFrame({**metrics_dict}, index=[0])
        return metrics_df


############
# Scoring
############


class Scores:
    @staticmethod
    def tp_score(y_true: np.ndarray, y_pred: np.ndarray,
                 target_label: Union[str, int], neu_label: int) -> np.ndarray:

        if target_label == 'all':
            return ((y_pred == y_true) & (y_pred != neu_label)).astype(np.int8)
        else:
            return ((y_pred == target_label) & (y_true == target_label)).astype(np.int8)

    @staticmethod
    def fp_score(y_true: np.ndarray, y_pred: np.ndarray,
                 target_label: Union[str, int], neu_label: int) -> np.ndarray:

        if target_label == 'all':
            return ((y_pred != y_true) & (y_pred != neu_label)).astype(np.int8)
        else:
            return ((y_pred == target_label) & (y_true != target_label)).astype(np.int8)

    @staticmethod
    def tn_score(y_true: np.ndarray, y_pred: np.ndarray,
                 target_label: Union[str, int], neu_label: int) -> np.ndarray:
        if target_label == 'all':
            return ((y_pred == neu_label) & (y_true == neu_label)).astype(np.int8)
        else:
            return ((y_pred != target_label) & (y_true != target_label)).astype(np.int8)

    @staticmethod
    def fn_score(y_true: np.ndarray, y_pred: np.ndarray,
                 target_label: Union[str, int], neu_label: int) -> np.ndarray:
        if target_label == 'all':
            return ((y_pred == neu_label) & (y_true != neu_label)).astype(np.int8)
        else:
            return ((y_pred != target_label) & (y_true == target_label)).astype(np.int8)

    @classmethod
    def n_tp_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                   target_label: Union[str, int], neu_label: int) -> int:

        return cls.tp_score(y_true, y_pred, target_label, neu_label).sum()

    @classmethod
    def n_fp_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                   target_label: Union[str, int], neu_label: int) -> int:

        return cls.fp_score(y_true, y_pred, target_label, neu_label).sum()

    @classmethod
    def n_tn_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                   target_label: Union[str, int], neu_label: int) -> int:
        return cls.tn_score(y_true, y_pred, target_label, neu_label).sum()

    @classmethod
    def n_fn_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                   target_label: Union[str, int], neu_label: int) -> int:
        return cls.fn_score(y_true, y_pred, target_label, neu_label).sum()

    @classmethod
    def positives_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                        target_label: Union[str, int], neu_label: int) -> np.ndarray:
        tp = cls.tp_score(y_true, y_pred, target_label, neu_label)
        fp = cls.fp_score(y_true, y_pred, target_label, neu_label)
        return tp - fp

    @classmethod
    def negatives_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                        target_label: Union[str, int], neu_label: int) -> np.ndarray:

        tn = cls.tn_score(y_true, y_pred, target_label, neu_label)
        fn = cls.fn_score(y_true, y_pred, target_label, neu_label)
        return tn - fn

    @classmethod
    def n_positives_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                          target_label: Union[str, int], neu_label: int) -> int:

        return np.count_nonzero(cls.positives_score(y_true, y_pred, target_label, neu_label))

    @classmethod
    def n_negatives_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                          target_label: Union[str, int], neu_label: int) -> int:

        return np.count_nonzero(cls.negatives_score(y_true, y_pred, target_label, neu_label))

    @classmethod
    def accuracy_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                       target_label: Union[str, int], neu_label: int) -> np.ndarray:

        return (y_true == y_pred).sum() / y_true.size

    @classmethod
    def precision_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                        target_label: Union[str, int], neu_label: int) -> float:

        tp = cls.n_tp_score(y_true, y_pred, target_label, neu_label)
        fp = cls.n_fp_score(y_true, y_pred, target_label, neu_label)
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    @classmethod
    def recall_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                     target_label: Union[str, int], neu_label: int) -> float:

        tp = cls.n_tp_score(y_true, y_pred, target_label, neu_label)
        fn = cls.n_fn_score(y_true, y_pred, target_label, neu_label)
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    @classmethod
    def activity_score(cls, y_true: np.ndarray, y_pred: np.ndarray,
                       target_label: Union[str, int], neu_label: int) -> float:

        return cls.n_positives_score(y_true, y_pred, target_label, neu_label) / y_true.size
