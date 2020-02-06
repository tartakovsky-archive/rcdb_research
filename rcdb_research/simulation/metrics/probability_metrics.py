from typing import Tuple

import numpy as np

from ..entities import Probabilities
from sklearn.calibration import calibration_curve


class ProbabilityMetrics:
    """
    Class for analyzing Probabilities objects outputted by ML models
    """
    ############
    # Initialization
    ############

    def __init__(self, probas: 'Probabilities'):
        """
        :param probas: instance of Probabilities class
        """
        self.probas = probas

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
