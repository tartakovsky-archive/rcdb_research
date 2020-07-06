import numpy_ext as npext

from ..features import misc
from ..job_manager import km


diff_step = npext.expstep_range(1, 50, 1, step_mult=0.1)
window = npext.expstep_range(1, 100, 1, step_mult=0.03)

price_config = dict(
    price=[
        dict(
            alias='change',
            fn=misc.frac_change,
            pg=km(step=diff_step),
            dm=km(series=['close']),
        ),
        dict(
            alias='detrended',
            fn=misc.series_ma_frac_change,
            pg=km(window=window),
            dm=km(series=['close']),
        ),
    ]
)
