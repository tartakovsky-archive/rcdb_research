import numpy_ext as npext

from ..features import misc
from ..job_manager import km, t


diff_step = npext.expstep_range(1, 50, 1, step_mult=0.1)
window = npext.expstep_range(1, 100, 1, step_mult=0.03)

timediff_config = dict(
    timediff=[
        dict(
            alias='itself',
            fn=misc.diff,
            pg=km(step=diff_step),
            dm=km(series=['timestamp']),
            tr=[t.symlog()]
        ),
        dict(
            alias='detrended',
            fn=misc.series_ma_frac_change,
            pg=km(window=window),
            dm=km(series=[km.col("timediff").t([t.symlog()])]),
        ),
    ]
)
