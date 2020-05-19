import numpy as np
import scipy.stats as ss
from sklearn.metrics import mutual_info_score


def _pairwise_variation_of_information(x: np.ndarray, y: np.ndarray, bins: int, normalized: bool, method: str) -> np.ndarray:
    """
    Sources: 
        1. MACHINE LEARNING FOR ASSET MANAGERS, Marcos M. López de Prado, ISBN 978-1-108-79289-9, p. 44
        2. MACHINE LEARNING FOR ASSET MANAGERS, Marcos M. López de Prado, ISBN 978-1-108-79289-9, p. 46
        3. Information Theoretic Measures for Clusterings Comparison, Vinh et al, doi: 10.1145/1553374.1553511
    :param x:
    :param y:
    :param bins:
    :param normalized:
    :return:
    """
    supported_methods = ['max', 'joint']
    if method not in supported_methods:
        raise ValueError(f'method {method} not in supported: {supported_methods}')
    
    # short-circuiting
    if (x == y).all():
        return 0

    def optimal_bins(x: np.ndarray, y: np.ndarray) -> int:
        corr = np.corrcoef(x, y)[0, 1]
        nObs = (x.shape[0] + y.shape[0]) // 2
        b=round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**.5)**.5)
        if np.isnan(b) or not np.isfinite(b):
            warnings.warn(f'optimal bins size not computed (b={b}), defaulting to b=20')
            b = 20
        return int(b)
    
    if bins is None:
        bins = optimal_bins(x, y)
    
    cXY = np.histogram2d(x, y, bins)[0]
    iXY = mutual_info_score(None, None, contingency=cXY)
    hX = ss.entropy(np.histogram(x, bins)[0])  # marginal
    hY = ss.entropy(np.histogram(y, bins)[0])  # marginal
    
    numerator = {
        'max': max(hX, hY) - iXY,
        'joint': hX + hY - 2*iXY
    }.get(method)
    
    denominator = {
        'max': max(hX, hY),
        'joint': hX + hY - iXY
    }.get(method)
    
    if normalized:
        return numerator / denominator
    return numerator


def nid(x, y):
    return _pairwise_variation_of_information(x, y, None, True, 'max')
