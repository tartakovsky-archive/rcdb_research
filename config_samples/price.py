from . import utils
from . import osc
from rcdb_research.features.parallel_calc_all import km

diff_step = utils.pct_range(1, 50, 1, mult_step=0.1)
window = utils.pct_range(1, 100, 1, mult_step=0.03)

price_config = dict(
    price=[
        dict(
            alias='change',
            fn=osc.frac_diff,
            pg=km(step=diff_step),
            dm=km(series=['close']),
        ),
        dict(
            alias='detrended',
            fn=osc.series_ma_frac_diff,
            pg=km(window=window),
            dm=km(series=['close']),
        ),
    ]
)
