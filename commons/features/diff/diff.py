import numpy as np
from tulipindicators import ti

def diff(series, step=1, fillna=np.nan):
    r = np.empty(series.size)
    r.fill(fillna)
    r[step:] = series[step:] - series[:-step]
    return r

def frac_diff(series, step=1, fillna=np.nan):
    return np.hstack((
        [fillna for _ in range(step)],
        np.divide((series[step:] - series[:-step]), series[:-step])
    ))

def series_ma_frac_diff(series, window, minus=1):
    return np.divide(series, ti.sma(series, window)) - minus

def two_series_ma_frac_diff(series1, series2, window, minus=1):
    return np.divide(ti.sma(series1, window), ti.sma(series2, window)) - minus
