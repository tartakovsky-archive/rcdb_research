from typing import Tuple
import weakref

import numpy as np
import scipy.optimize as opt

from ..entities import Probabilities
from ..simulators import PredictionSimulator

from sklearn.calibration import calibration_curve


class ProbabilityMetrics:
    """
    Class for analyzing Probabilities objects outputted by ML models
    """
    ############
    # Initialization
    ############

    def __init__(self, probas: 'Probabilities', labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}):
        """
        :param probas: instance of Probabilities class
        """
        self.probas = weakref.proxy(probas)
        self.labels = labels

    def calibration(self, n_bins=40, strategy='uniform', normalize=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper over sklearn.calibration.calibration_curve

        For more details see:
        https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html

        :param normalize: Whether y_prob needs to be normalized into the bin [0, 1]
        :param n_bins: Number of bins
        :param strategy: Strategy used to define the widths of the bins.

        :return: tupel of (proba_true: np.ndarray, proba_pred: np.ndarray).
            The true probability in each bin (fraction of positives) and the mean predicted probability in each bin.
        """
        return calibration_curve(
            self.probas.y_true,
            self.probas.y_pred_proba,
            normalize=normalize,
            n_bins=n_bins,
            strategy=strategy
        )

    def threshold_for_precision(self, probas: 'Probabilities', target: float,
                                direction: str = 'pos', tolerance=1e-3) -> float:
        """
        Uses secant root finding method to search for threshold value that produces target precision
        :param probas: instance of Probabilities class
        :param target: target value of the metric
        :param direction: one of ['pos', 'neg']
        :param tolerance: maximum allowed error of metric value
        :returns: threshold value which produces target value of the metric
        """
        supported_direction = ['pos', 'neg']
        if direction not in supported_direction:
            raise ValueError(f'direction should be one of {supported_direction}')

        preds_sim = PredictionSimulator(self.labels)

        def f(threshold):
            preds = preds_sim.pos_preds(probas, threshold).init_metrics(target_label=self.labels[direction])
            return preds.metrics.precision() - target

        return opt.root_scalar(f, method='secant', x0=0.5, x1=0.55, xtol=tolerance, maxiter=1000).root

    def threshold_for_activity(self, probas: 'Probabilities', target: float,
                               direction: str = 'pos', tolerance=1e-5) -> float:
        """
        Uses brentq root finding method to search for threshold value that produces target activity
        :param probas: instance of Probabilities class
        :param target: target value of the metric
        :param direction: one of ['pos', 'neg']
        :param tolerance: maximum allowed error of metric value
        :returns: threshold value which produces target value of the metric
        """
        supported_direction = ['pos', 'neg']
        if direction not in supported_direction:
            raise ValueError(f'direction should be one of {supported_direction}')

        preds_sim = PredictionSimulator(self.labels)

        def f(threshold):
            preds = preds_sim.pos_preds(probas, threshold).init_metrics(target_label=self.labels[direction])
            return preds.metrics.activity() - target

        return opt.brentq(f, a=0, b=1, xtol=tolerance, maxiter=1000)
