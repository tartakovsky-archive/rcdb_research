from typing import Callable

import numpy as np
from sklearn.metrics import log_loss, make_scorer

from ..simulation.probas import Probabilities
from ..models import NoSkillClassifier


def neg_log_loss(y_true: np.ndarray, y_proba: np.ndarray, sample_weight=None) -> float:
    return -1 * log_loss(y_true, y_proba, sample_weight=sample_weight)


neg_log_loss_scorer = make_scorer(neg_log_loss, needs_proba=True)


def bounded_log_loss(y_true: np.ndarray,
                     y_proba: np.ndarray,
                     bounds=(0.692, 0.684),
                     sample_weight=None) -> float:
    logloss = log_loss(y_true, y_proba, sample_weight=sample_weight)
    return (logloss - bounds[0]) / (bounds[1] - bounds[0])


bounded_log_loss_scorer = make_scorer(bounded_log_loss, needs_proba=True)

##################
# Bounded Relative Log Loss
##################

def bounded_relative_log_loss(y_true, y_proba, no_skill_score=None, bounds=(0.0015, 0.013), sample_weight=None):
    if no_skill_score is None:
        X = np.zeros((y_true.size, 2))
        ns_y_proba = NoSkillClassifier('log_loss').fit(X, y_true).predict_proba(X)
        no_skill_score = log_loss(y_true, ns_y_proba)

    raw_score = log_loss(y_true, y_proba, sample_weight=sample_weight)
    relative_score = 1 - raw_score / no_skill_score
    bounded_score = (relative_score - bounds[0]) / (bounds[1] - bounds[0])
    return bounded_score


bounded_relative_log_loss_scorer = make_scorer(bounded_relative_log_loss, needs_proba=True)
brll = bounded_relative_log_loss
brll_scorer = bounded_relative_log_loss_scorer

##################


def precision_score_for_activity(activity: float,
                                 direction: str = 'pos',
                                 tolerance=1e-5) -> Callable[[np.ndarray, np.ndarray], float]:
    def precision_for_activity(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        pr = Probabilities(
            y_true=y_true,
            y_pred_proba=y_proba
        )
        return pr.metrics.precision_for_activity(
            target=activity, direction=direction, tolerance=tolerance
        )

    return precision_for_activity


def calibration(y_true: np.ndarray,
                y_proba: np.ndarray) -> tuple:
    bins = np.linspace(0, 1, 25)
    binids = np.digitize(y_proba, bins) - 1

    true_probas = []
    stds = []
    pred_probas = []
    for binid in np.unique(binids):
        select = binids == binid
        data = np.hstack(y_true[select])
        true_probas.append(data.mean())
        stds.append(data.std())
        pred_probas.append(y_proba[select].mean())
    true_probas = np.array(true_probas)
    stds = np.array(stds)
    pred_probas = np.array(pred_probas)
    return true_probas, stds, pred_probas
