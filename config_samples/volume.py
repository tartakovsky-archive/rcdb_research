from . import utils
from . import osc
from commons.features.parallel_calc_all import km, t

diff_step = utils.pct_range(1, 50, 1, mult_step=0.1)
window = utils.pct_range(1, 100, 1, mult_step=0.03)

volume_config = dict(
    volume=[
        dict(
            alias='detrended',
            fn=osc.series_ma_frac_diff,
            pg=km(window=window),
            dm=km(series=[km.col("volume").t([t.symlog()])]),
        ),
        dict(
            alias='buy_fraction',
            fn=osc.two_series_ma_frac_diff,
            pg=km(window=window, minus=[0.5]),
            dm=km(series1=['volume_buy'], series2=['volume']),
        ),
        dict(
            alias='sell_fraction',
            fn=osc.two_series_ma_frac_diff,
            pg=km(window=window, minus=[0.5]),
            dm=km(series1=['volume_sell'], series2=['volume']),
        ),
        dict(
            alias='imbalance',
            fn=osc.two_series_ma_frac_diff,
            pg=km(window=window),
            dm=km(series1=['volume_buy'], series2=['volume_sell']),
        ),
    ]
)
