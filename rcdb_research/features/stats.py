import numpy as np
import numpy_ext as npext
from scipy import stats as st


def cmean(series: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the circular mean.

    Parameters
    ----------
    series : np.ndarray
        Input series
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Result array
    """
    return st.circmean(npext.rolling(series, window, as_array=True), axis=1)


def fkurtosis(series: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the Fisher kurtosis

    Parameters
    ----------
    series : np.ndarray
        Input series
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Result array
    """
    return st.kurtosis(npext.rolling(series, window, as_array=True), axis=1)


def pkurtosis(series: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the Pearson kurtosis

    Parameters
    ----------
    series : np.ndarray
        Input series
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Result array
    """
    return st.kurtosis(npext.rolling(series, window, as_array=True), fisher=False, axis=1)


def skewness(series: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the sample skewness

    Parameters
    ----------
    series : np.ndarray
        Input series
    window : int
        Rolling window size

    Returns
    -------
    np.ndarray
        Result array
    """
    return st.skew(npext.rolling(series, window, as_array=True), axis=1)
