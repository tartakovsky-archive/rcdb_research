from typing import Union
from math import factorial, log

import numpy as np
import numpy_ext as npext
from numba import jit
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch


def app_entropy(series: np.ndarray, window: int, order: int = 2, metric: str = 'chebyshev') -> np.ndarray:
    """
    Approximate Entropy

    Parameters
    ----------
    series : np.ndarray
        Input data
    window : int
        Rolling window size
    order : int
        Embedding dimension
    metric : str, optional
        metrics name for sklearn.neighbors.KDTree,
        list of metrics available at KDTree.valid_metrics

    Returns
    -------
    np.ndarray
        Approximate Entropy series
    """

    def _app_entropy(x, order=2, metric='chebyshev'):
        phi = _app_samp_entropy(x, order=order, metric=metric, approximate=True)
        return np.subtract(phi[0], phi[1])

    return npext.prepend_na(
        [
            _app_entropy(s, order=order, metric=metric)
            for s in npext.rolling(series, window, as_array=True)[window - 1:]
        ],
        window - 1
    )


def sample_entropy(series: np.ndarray, window: int, order: int = 2, metric: str = 'chebyshev') -> np.ndarray:
    """
    Sample Entropy

    Parameters
    ----------
    series : np.ndarray
        Input data
    window : int
        Rolling window size
    order : int
        Embedding dimension
    metric : str, optional
        metrics name for sklearn.neighbors.KDTree,
        list of metrics available at KDTree.valid_metrics

    Returns
    -------
    np.ndarray
        Sample Entropy series
    """

    def _sample_entropy(x, order=2, metric='chebyshev'):
        x = np.asarray(x, dtype=np.float64)
        if metric == 'chebyshev' and x.size < 5000:
            return _numba_sampen(x, mm=order, r=0.2)
        else:
            phi = _app_samp_entropy(x, order=order, metric=metric,
                                    approximate=False)
            return -np.log(np.divide(phi[1], phi[0]))

    res = npext.rolling_apply(
        _sample_entropy,
        window,
        series,
        order=order,
        metric=metric
    )
    res[np.isnan(res)] = 0
    return res


def spectral_entropy(
    series: np.ndarray,
    window: int,
    sf: float,
    method: str = 'fft',
    nperseg: Union[str, int] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Spectral Entropy

    Parameters
    ----------
    series : np.ndarray
        Input data
    window : int
        Rolling window size
    sf : float
        Sampling frequency
    method : str, optional
        Spectral estimation method:
        - 'fft' : Fourier Transform (via scipy.signal.periodogram) (default)
        - 'welch' : Welch periodogram (via scipy.signal.welch))
    nperseg : Union[str, int], optional
        Length of each FFT segment for Welch method.
        If None (default), uses scipy default of 256 samples
    normalize : bool, optional
        if True, divide by log2(psd.size) to normalize the
        spectral entropy between 0 and 1. Otherwise,
        return the spectral entropy in bit (default).

    Returns
    -------
    np.ndarray
        Spectral Entropy series
    """

    def _spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False):
        x = np.array(x)
        # Compute and normalize power spectrum
        if method == 'fft':
            _, psd = periodogram(x, sf)
        elif method == 'welch':
            _, psd = welch(x, sf, nperseg=nperseg)
        psd_norm = np.divide(psd, psd.sum())
        se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
        if normalize:
            se /= np.log2(psd_norm.size)
        return se

    res = npext.rolling_apply(
        _spectral_entropy,
        window,
        series,
        sf=sf,
        method=method,
        nperseg=nperseg,
        normalize=normalize
    )
    res[np.isnan(res)] = 0
    return res


def svd_entropy(series: np.ndarray, window: int, order: int = 3, delay: int = 1, normalize: bool = False) -> np.ndarray:
    """
    Singular Value Decomposition entropy

    Parameters
    ----------
    series : np.ndarray
        Input data
    window : int
        Rolling window size
    order : int, optional
        Order of permutation entropy
    delay : int, optional
        Time delay
    normalize : bool, optional
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    np.ndarray
        Singular Value Decomposition entropy series
    """

    def _svd_entropy(x, order=3, delay=1, normalize=False):
        x = np.array(x)
        mat = _embed(x, order=order, delay=delay)
        W = np.linalg.svd(mat, compute_uv=False)
        # Normalize the singular values
        W /= sum(W)
        svd_e = -np.multiply(W, np.log2(W)).sum()
        if normalize:
            svd_e /= np.log2(order)
        return svd_e

    res = npext.rolling_apply(
        _svd_entropy,
        window,
        series,
        order=order,
        delay=delay,
        normalize=normalize
    )
    res[np.isnan(res)] = 0
    return res


def perm_entropy(
        series: np.ndarray,
        window: int,
        order: int = 3,
        delay: int = 1,
        normalize: bool = False) -> np.ndarray:
    """
    Permutation Entropy

    Parameters
    ----------
    series : np.ndarray
        Input data
    window : int
        Rolling window size
    order : int, optional
        Order of permutation entropy. Default is 3
    delay : int, optional
        Time delay. Default is 1
    normalize : bool, optional
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise (default), return the permutation entropy in bit.
    Returns
    -------
    np.ndarray
        Permutation Entropy series
    """

    def _perm_entropy(x, order=3, delay=1, normalize=False):
        x = np.array(x)
        ran_order = range(order)
        hashmult = np.power(order, ran_order)
        # Embed x and sort the order of permutations
        sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
        # Associate unique integer to each permutations
        hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
        # Return the counts
        _, c = np.unique(hashval, return_counts=True)
        # Use np.true_divide for Python 2 compatibility
        p = np.true_divide(c, c.sum())
        pe = -np.multiply(p, np.log2(p)).sum()
        if normalize:
            pe /= np.log2(factorial(order))
        return pe

    return npext.rolling_apply(
        _perm_entropy,
        window,
        series,
        order=order,
        delay=delay,
        normalize=normalize
    )


def binned_entropy(series: np.ndarray, window: int, max_bins: int) -> np.ndarray:
    """
    First bins the values of x into max_bins equidistant bins in each window

    Parameters
    ----------
     series : np.ndarray
        Input data
    window : int
        Rolling window size
    max_bins : int
        The maximal number of bins
    Returns
    -------
    np.ndarray
        Result series
    """
    res = [
        -np.sum([p * np.math.log(p) for p in (np.histogram(x, bins=max_bins)[0] / x.size) if p != 0])
        for x in npext.rolling(series, window, as_array=True)[window - 1:]
    ]

    return np.hstack(([np.nan for _ in range(window - 1)], res))


###################
# Entropy helpers
###################


def _embed(x, order=3, delay=1):
    """Time-delay embedding.
    Parameters
    ----------
    x : 1d-array, shape (n_times)
        Time series
    order : int
        Embedding dimension (order)
    delay : int
        Delay.
    Returns
    -------
    embedded : ndarray, shape (n_times - (order - 1) * delay, order)
        Embedded time-series.
    """
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T


def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
    """
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    phi = np.zeros(2)
    r = 0.2 * np.std(x, axis=-1, ddof=1)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


@jit('f8(f8[:], i4, f8)', nopython=True)
def _numba_sampen(x, mm=2, r=0.2):
    """
    Fast evaluation of the sample entropy using Numba.
    """
    n = x.size
    n1 = n - 1
    mm += 1
    mm_dbld = 2 * mm

    # Define threshold
    r *= x.std()

    # initialize the lists
    run = [0] * n
    run1 = run[:]
    r1 = [0] * (n * mm_dbld)
    a = [0] * mm
    b = a[:]

    for i in range(n1):
        nj = n1 - i

        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1:
                        b[m] += 1
            else:
                run[jj] = 0
        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    m = mm - 1

    while m > 0:
        b[m] = b[m - 1]
        m -= 1

    b[0] = n * n1 / 2
    p = np.true_divide(
        np.array(a, dtype=np.float64),
        np.array(b, dtype=np.float64)
    )
    return -log(p[-1])
