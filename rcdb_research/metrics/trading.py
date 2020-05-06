import numpy as np


def equity(returns: np.ndarray, initial_equity: float = 100) -> np.ndarray:
    e = initial_equity
    res = np.empty(returns.shape)
    for i in range(len(res)):
        e += e * returns[i]
        res[i] = e

    return res


def cum_return(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(returns + 1) - 1


def drawdown(cum_return: np.ndarray) -> np.ndarray:
    x = cum_return + 1
    equity = x * x[0]
    return equity / np.maximum.accumulate(equity) - 1


def abs_return(cum_return: np.ndarray) -> float:
    return cum_return[-1]


def cagr(abs_return: float, years_passed: float) -> float:
    return (1 + abs_return) ** (1 / years_passed) - 1


def mar(cagr: float, max_dd: float) -> float:
    return cagr / np.abs(max_dd)
