import scipy
import numpy as np
import numpy_ext as npext


###########
# Helpers
###########

def _rolling_means(price: np.ndarray, min_window: int, max_window: int, number_of_rollings: int) -> np.ndarray:
    """
    Calculate rollings windows for price and cut nans part

    Parameters
    ----------
    price : np.ndarray
        Asset price array
    min_window : int
        Minimum rolling window size
    max_window : int
        Maximum rolling window size
    number_of_rollings : int
        Count of rolling windows

    Returns
    -------
    np.ndarray

    """
    rollings = []
    for window in np.unique(np.linspace(min_window, max_window, number_of_rollings).astype(int)):
        rollings.append(
            np.mean(npext.rolling(price, int(window), as_array=True), axis=1)
        )
    return np.array(rollings)[:, max_window - 1:].transpose()


###########
# Features
###########

def distance(
        price: np.ndarray,
        min_window: int = 10,
        max_window: int = 100,
        number_of_rollings: int = 10,
        hamming_v: np.ndarray = np.arange(1, 11)
) -> np.ndarray:
    """
    Calculates distance metric

    Parameters
    ----------
    price : np.ndarray
        Asset price array
    min_window : int
        Minimum rolling window size
    max_window : int
        Maximum rolling window size
    number_of_rollings : int
        Count of rolling windows
    hamming_v : np.ndarray
        v parameter for scipy.spatial.distance.hamming function

    Returns
    -------
    np.ndarray
        distance metric

    """

    res = np.apply_along_axis(
        func1d=lambda mavg: scipy.spatial.distance.hamming(
            scipy.stats.rankdata(mavg), hamming_v
        ),
        axis=1,
        arr=_rolling_means(price, min_window, max_window, number_of_rollings)
    )
    return npext.prepend_na(res, max_window - 1)


def correlation(
        price: np.ndarray,
        min_window: int = 10,
        max_window: int = 100,
        number_of_rollings: int = 10,
        spearmanr_b: np.ndarray = np.arange(1, 11)
) -> np.ndarray:
    """
    Calculates correlation metric

    Parameters
    ----------
    price : np.ndarray
        Asset price array
    min_window : int
        Minimum rolling window size
    max_window : int
        Maximum rolling window size
    number_of_rollings : int
        Count of rolling windows
    spearmanr_b : np.ndarray
        b parameter for scipy.stats.spearmanr function

    Returns
    -------
    np.ndarray
        correlation metric

    """
    res = np.apply_along_axis(
        func1d=lambda mavg: scipy.stats.spearmanr(
            scipy.stats.rankdata(mavg), spearmanr_b
        )[0],
        axis=1,
        arr=_rolling_means(price, min_window, max_window, number_of_rollings)
    )
    return npext.prepend_na(res, max_window - 1)


def thickness(
        price: np.ndarray,
        min_window: int = 10,
        max_window: int = 100,
        number_of_rollings: int = 10
) -> np.ndarray:
    """
    Calculates thickness metric

    Parameters
    ----------
     price : np.ndarray
        Asset price array
    min_window : int
        Minimum rolling window size
    max_window : int
        Maximum rolling window size
    number_of_rollings : int
        Count of rolling windows

    Returns
    -------
    np.ndarray:
        thikness metric
    """

    res = np.apply_along_axis(
        func1d=lambda mavg: np.max(mavg) - np.min(mavg),
        axis=1,
        arr=_rolling_means(price, min_window, max_window, number_of_rollings)
    )
    return npext.prepend_na(res, max_window - 1)


def p0(price: np.ndarray, k: int = 30):
    """
    Calculates measure of Momentum From Physics (p0)

    Parameters
    ----------
    price : np.ndarray
        Asset price
    k : int
        Rolling window size

    Returns
    -------
    np.ndarray
        measure of Momentum From Physics (p0)
    """

    v = npext.prepend_na(
        np.diff(
            np.log(price)
        ),
        1
    )
    return np.sum(
        npext.rolling(v, window=k, as_array=True),
        axis=1
    )


def p1(price: np.ndarray, volume: np.ndarray, k: int = 30):
    """
    Calculates measure of Momentum From Physics (p1)

    Parameters
    ----------
    price : np.ndarray
        Asset price
    volume : np.ndarray
        Asset volume
    k : int
        Rolling window size

    Returns
    -------
    np.ndarray
        measure of Momentum From Physics (p1)
    """

    return volume * p0(price, k)


def p2(price: np.ndarray, volume: np.ndarray, k: int = 30):
    """
    Calculates measure of Momentum From Physics (p2)

    Parameters
    ----------
    price : np.ndarray
        Asset price
    volume : np.ndarray
        Asset volume
    k : int
        Rolling window size

    Returns
    -------
    np.ndarray
        measure of Momentum From Physics (p2)
    """

    return p1(price, volume, k) / np.sum(npext.rolling(volume, window=k, as_array=True), axis=1)


def p3(price: np.ndarray, k: int = 30):
    """
    Calculates measure of Momentum From Physics (p3)

    Parameters
    ----------
    price : np.ndarray
        Asset price
    k : int
        Rolling window size

    Returns
    -------
    np.ndarray
        measure of Momentum From Physics (p3)
    """

    v = npext.prepend_na(
        np.diff(
            np.log(price)
        ),
        1
    )
    v_rolling = npext.rolling(v, window=k, as_array=True)
    return np.mean(v_rolling, axis=1) / np.std(v_rolling, axis=1)
