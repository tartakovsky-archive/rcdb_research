import numba  # noqa

import scipy.optimize as opt
import numpy as np

from ..simulation import Probabilities, PredictionSimulator


def threshold_for_metric(probas: 'Probabilities', metric: str, target: float, tolerance=0.001) -> float:
    """
    Uses newton root finding method to search for threshold value that produces target value of the metric
    :param probas: instance of Probabilities class
    :param metric: one of ['activity', 'precision']
    :param target: target value of the metric
    :param tolerance: maximum allowed error of metric value
    :returns: threshold value which produces target value of the metric
    """

    supported_metrics = ['precision', 'activity']

    if metric not in supported_metrics:
        raise ValueError(f'metric should be one of {supported_metrics}')

    preds_sim = PredictionSimulator()

    def f(threshold):
        preds = preds_sim.pos_preds(probas, threshold)

        if metric == 'precision':
            return preds.precision() - target
        elif metric == 'activity':
            return preds.activity() - target
        else:
            return 0

    return opt.newton(f, 0.5, tol=tolerance, maxiter=1000)


@numba.jit
def symlog(x, C=0):
    return np.sign(x) * (np.log2(1 + np.abs(x) / (10 ** C)))


def symscale(x: np.array, center: float = 0, to_range: tuple = (-1, 0, 1)) -> np.array:
    x_above = x[x >= center]
    x_below = x[x <= center]

    scaled_above = np.interp(x_above, (x_above.min(), x_above.max()), (to_range[1], to_range[2]))
    scaled_below = np.interp(x_below, (x_below.min(), x_below.max()), (to_range[0], to_range[1]))

    x_scaled = np.zeros(x.size)
    x_scaled[x >= center] = scaled_above
    x_scaled[x <= center] = scaled_below
    x_scaled[x == center] = to_range[1]

    return x_scaled
