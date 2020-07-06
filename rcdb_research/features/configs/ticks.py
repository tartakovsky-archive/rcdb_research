import numpy_ext as npext

from ..features import misc
from ..job_manager import km, t


diff_step = npext.expstep_range(1, 50, 1, step_mult=0.1)
window = npext.expstep_range(1, 100, 1, step_mult=0.03)

ticks_config = dict(
    ticks=[
        dict(
            alias='detrended',
            fn=misc.series_ma_frac_change,
            pg=km(window=window),
            dm=km(series=[km.col("ticks").t([t.symlog()])]),
        ),
        dict(
            alias='buy_fraction',
            fn=misc.two_series_ma_frac_change,
            pg=km(window=window, minus=[0.5]),
            dm=km(series1=['ticks_buy'], series2=['ticks']),
        ),
        dict(
            alias='sell_fraction',
            fn=misc.two_series_ma_frac_change,
            pg=km(window=window, minus=[0.5]),
            dm=km(series1=['ticks_sell'], series2=['ticks']),
        ),
        dict(
            alias="imbalance",
            fn=misc.two_series_ma_frac_change,
            pg=km(window=window),
            dm=km(series1=['ticks_buy'], series2=['ticks_sell']),
        ),
    ]
)
