import logging

import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score


def nmi(x, y) -> float:
    """
    Source:
    Information Theoretic Measures for Clusterings Comparison, Vinh et al, doi: 10.1145/1553374.1553511

    :param x: np.ndarray
    :param y: np.ndarray
    :return: Normalized Mutual Information (method: max)
    """
    return mutual_info(x, y, normalized=True, method='max')


def optimal_bins(x: np.ndarray, y: np.ndarray) -> int:
    corr = np.corrcoef(x, y)[0, 1]
    nObs = (x.shape[0] + y.shape[0]) // 2
    b = round(2 ** -.5 * (1 + (1 + 24 * nObs / (1. - corr ** 2)) ** .5) ** .5)
    if np.isnan(b) or not np.isfinite(b):
        logging.warning(f'optimal bins size not computed (b={b}), defaulting to b=20')
        b = 20
    return int(b)


def mutual_info(x, y, bins=None, normalized=False, method='joint'):
    if bins is None:
        bins = optimal_bins(x, y)

    cXY = np.histogram2d(x, y, bins)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    hX = ss.entropy(np.histogram(x, bins)[0])  # marginal
    hY = ss.entropy(np.histogram(y, bins)[0])  # marginal
    hXY = hX + hY - iXY

    numerator = {
        'max': iXY,
        'joint': iXY
    }.get(method)

    denominator = {
        'max': max(hX, hY),
        'joint': hXY
    }.get(method)

    if normalized:
        return numerator / denominator

    return numerator
