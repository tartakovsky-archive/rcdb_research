import numpy as np
from fnlib import stats

from . import utils
from commons.features.parallel_calc_all import km


window = np.array(utils.pct_range(2, 100, 1, mult_step=0.03)).astype(int).tolist()
stats_conf = dict(
    stats=[
        dict(
            fn=stats.cmean,
            pg=km(window=window),
            dm=km(series=['close', 'timediff', 'volume_quote', 'ticks']),
        ),
        dict(
            fn=stats.fkurtosis,
            pg=km(window=window),
            dm=km(series=['close', 'timediff', 'volume_quote', 'ticks']),
        ),
        dict(
            fn=stats.pkurtosis,
            pg=km(window=window),
            dm=km(series=['close', 'timediff', 'volume_quote', 'ticks']),
        ),
        dict(
            fn=stats.skewness,
            pg=km(window=window),
            dm=km(series=['close', 'timediff', 'volume_quote', 'ticks']),
        ),
    ]
)
