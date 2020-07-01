import numpy as np
import numpy_ext as npext

from ..features import stats
from ..job_manager import km


window = np.array(npext.expstep_range(2, 100, 1, step_mult=0.03)).astype(int).tolist()
stats_config = dict(
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
