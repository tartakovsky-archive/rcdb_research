from typing import Callable

import numpy as np
from sklearn.metrics import log_loss

from ..simulation.probas import Probabilities
from ..utils import probabilities_to_predictions


def neg_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Negative LogLoss
    :param y_true: trues values
    :param y_pred: predicted values
    :return:
    """
    return -1 * log_loss(y_true, y_pred)


def pwa(y_true: np.ndarray, y_proba: np.ndarray, labels=(-1, 1)) -> float:
    """
    Calculates Probability-weighted accuracy.
    For more details see: Lopez, Machine Learning for Asset Managers, 6.4

    :param y_true: trues values
    :param y_pred: predicted values
    :param y_proba: probabilities (n_sample, n_labels) or positive label (n_sample,)
    :return:
    """
    if len(y_proba.shape) == 1:
        y_proba = np.vstack((1 - y_proba, y_proba)).T

    y_pred = probabilities_to_predictions(y_proba, labels)

    num_labels = y_proba.shape[1]
    pn_minus_k = np.max(y_proba, axis=1) - num_labels ** -1

    correct_predict = (y_true == y_pred) * 1

    return np.sum(correct_predict * pn_minus_k) / np.sum(pn_minus_k)


def precision_score_for_activity(
    activity: float,
    direction: str = 'pos',
    tolerance=1e-5
) -> Callable[[np.ndarray, np.ndarray], float]:

    def precision_for_activity(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        pr = Probabilities(
            y_true=y_true,
            y_pred_proba=y_proba
        )
        return pr.metrics.precision_for_activity(
            target=activity, direction=direction, tolerance=tolerance
        )

    return precision_for_activity
