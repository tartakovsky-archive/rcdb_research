from . import utils
from . import osc
from rcdb_research.features.parallel_calc_all import km, t


diff_step = utils.pct_range(1, 50, 1, mult_step=0.1)
window = utils.pct_range(1, 100, 1, mult_step=0.03)

timediff_config = dict(
    timediff=[
        dict(
            alias='itself',
            fn=osc.diff,
            pg=km(step=diff_step),
            dm=km(series=['timestamp']),
            tr=[t.symlog()]
        ),
        dict(
            alias='detrended',
            fn=osc.series_ma_frac_diff,
            pg=km(window=window),
            dm=km(series=[km.col("timediff").t([t.symlog()])]),
        ),
    ]
)
