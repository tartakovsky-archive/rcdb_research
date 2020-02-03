import numba  # noqa

import scipy.optimize as opt
import numpy as np

from ..simulation import Probabilities, PredictionSimulator


def threshold_for_precision(probas: 'Probabilities', target: float, tolerance=0.001) -> float:
    """
    Uses secant root finding method to search for threshold value that produces target precision
    :param probas: instance of Probabilities class
    :param target: target value of the metric
    :param tolerance: maximum allowed error of metric value
    :returns: threshold value which produces target value of the metric
    """

    preds_sim = PredictionSimulator()

    def f(threshold):
        preds = preds_sim.pos_preds(probas, threshold)
        return preds.precision() - target

    return opt.root_scalar(f, method='secant', x0=0.5, x1=0.55, xtol=tolerance, maxiter=1000).root


def threshold_for_activity(probas: 'Probabilities', target: float, tolerance=0.001) -> float:
    """
    Uses brentq root finding method to search for threshold value that produces target activity
    :param probas: instance of Probabilities class
    :param target: target value of the metric
    :param tolerance: maximum allowed error of metric value
    :returns: threshold value which produces target value of the metric
    """

    preds_sim = PredictionSimulator()

    def f(threshold):
        preds = preds_sim.pos_preds(probas, threshold)
        return preds.activity() - target

    return opt.brentq(f, a=0, b=1, xtol=tolerance, maxiter=1000)


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
