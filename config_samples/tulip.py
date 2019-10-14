from . import utils
from fnlib import tulip
from rcdb_research.features.parallel_calc_all import km, t, col


window = utils.pct_range(3, 50, 10, mult_step=0.03)
short_period = utils.pct_range(2, 25, 5, mult_step=0.03)
long_period = utils.pct_range(5, 40, 5, mult_step=0.03)

wma_period = window
roc_shorter_period = short_period
roc_longer_period = long_period

signal_period = window

ema_period = [x + 1 for x in window]
lookback_period = ema_period
stddev_period = ema_period

tulip_config = dict(
    tulip=[
        dict(
            fn=tulip.adosc,
            pg=km(short_period=short_period, long_period=long_period),
            dm=km(
                high=[col('high').symlog()],
                low=[col('low').symlog()],
                close=[col('close').symlog()],
                volume=[col('volume').symlog()],
            ),
            cn='p.short_period < p.long_period'
        ),
        dict(
            fn=tulip.adx,
            pg=km(period=window),
            dm=km(high=['high'], low=['low']),
            tr=[t.symlog()]
        ),
        dict(
            fn=tulip.adxr,
            pg=km(period=window),
            dm=km(
                high=['high'],
                low=['low'],
            ),
            tr=[t.symlog()]
        ),
        dict(
            fn=tulip.ao,
            dm=km(
                high=[col('high').symlog()],
                low=[col('low').symlog()],
            ),
        ),
        dict(
            fn=tulip.apo,
            pg=km(
                short_period=short_period,
                long_period=long_period
            ),
            dm=km(
                series=[col("close").symlog()],
            ),
            cn='p.short_period < p.long_period'
        ),
        dict(
            fn=tulip.aroonosc,
            pg=km(period=window),
            dm=km(
                high=['high'],
                low=['low'],
            ),
        ),
        dict(
            fn=tulip.cci,
            pg=km(period=window),
            dm=km(
                high=['high'],
                low=['low'],
                close=['close']
            )
        ),
        dict(
            fn=tulip.cmf,
            pg=km(period=window),
            dm=km(
                high=['high'],
                low=['low'],
                close=['close'],
                volume=['volume']
            )
        ),
        dict(
            fn=tulip.cmo,
            pg=km(period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.copp,
            pg=km(wma_period=window, roc_shorter_period=roc_shorter_period, roc_longer_period=roc_longer_period),
            dm=km(series=['open', 'high', 'low', 'close', 'volume']),
            cn='p.roc_shorter_period < p.roc_longer_period'
        ),
        dict(
            fn=tulip.cvi,
            pg=km(period=window),
            dm=km(high=['high'], low=['low']),
            tr=[t.symlog_symlog()]
        ),
        dict(
            fn=tulip.dpo,
            pg=km(period=window),
            dm=km(
                series=[col(c).symlog() for c in ('open', 'high', 'low', 'close', 'volume')]
            )
        ),
        dict(
            fn=tulip.dx,
            pg=km(period=window),
            dm=km(high=['high'], low=['low'])
        ),
        dict(
            fn=tulip.emv,
            dm=km(high=['high'], low=['low'], volume=['volume']),
            tr=[t.symlog()]
        ),
        dict(
            fn=tulip.fisher,
            pg=km(period=window),
            dm=km(high=['high'], low=['low'])
        ),
        dict(
            fn=tulip.fisher_signal,
            pg=km(period=window),
            dm=km(high=['high'], low=['low'])
        ),
        dict(
            fn=tulip.fosc,
            pg=km(period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.kst_signal,
            pg=km(**{
                'roc1': [10], 'roc2': [15], 'roc3': [20], 'roc4': [30],
                'ma1': [10], 'ma2': [10], 'ma3': [10], 'ma4': [15]
            }),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.linregslope,
            pg=km(period=window),
            dm=km(
                series=[col(c).symlog() for c in ('open', 'high', 'low', 'close', 'volume')]
            )
        ),
        dict(
            fn=tulip.macd,
            pg=km(signal_period=signal_period, long_period=long_period, short_period=short_period),
            dm=km(
                series=[col(c).symlog() for c in ('open', 'high', 'low', 'close', 'volume')]
            ),
            cn='p.short_period < p.long_period'
        ),
        dict(
            fn=tulip.macd_signal,
            pg=km(signal_period=signal_period, long_period=long_period, short_period=short_period),
            dm=km(
                series=[col(c).symlog() for c in ('open', 'high', 'low', 'close', 'volume')]
            ),
            cn='p.short_period < p.long_period'
        ),
        dict(
            fn=tulip.marketfi,
            dm=km(high=['high'], low=['low'], volume=['volume']),
            tr=[t.symlog_symlog()]
        ),
        dict(
            fn=tulip.mass,
            pg=km(period=window),
            dm=km(high=[col('high').sympower2()], low=[col('low').sympower2()])
        ),
        dict(
            fn=tulip.md,
            pg=km(period=window),
            dm=km(
                series=[col(c).symlog() for c in ('open', 'high', 'low', 'close', 'volume')]
            )
        ),
        dict(
            fn=tulip.mfi,
            pg=km(period=window),
            dm=km(high=['high'], low=['low'], close=['close'], volume=['volume'])
        ),
        dict(
            fn=tulip.minus_di,
            pg=km(period=window),
            dm=km(high=['high'], low=['low'], close=['close'])
        ),
        dict(
            fn=tulip.minus_dm,
            pg=km(period=window),
            dm=km(high=[col('high').symlog()], low=[col('low').symlog()])
        ),
        dict(
            fn=tulip.msw_sine,
            pg=km(period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.natr,
            pg=km(period=window),
            dm=km(high=[col('high').symlog()], low=[col('low').symlog()], close=[col('close').symlog()])
        ),
        dict(
            fn=tulip.pfe,
            pg=km(period=window, ema_period=ema_period),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.plus_di,
            pg=km(period=window),
            dm=km(high=['high'], low=['low'], close=['close'])
        ),
        dict(
            fn=tulip.plus_dm,
            pg=km(period=window),
            dm=km(high=[col('high').symlog()], low=[col('low').symlog()])
        ),
        dict(
            fn=tulip.posc,
            pg=km(period=window, ema_period=ema_period),
            dm=km(high=['high'], low=['low'], close=['close'])
        ),
        dict(
            fn=tulip.ppo,
            pg=km(short_period=short_period, long_period=long_period),
            dm=km(series=['open', 'high', 'low', 'close', 'volume']),
            cn='p.short_period < p.long_period'
        ),
        dict(
            fn=tulip.qstick,
            pg=km(period=window),
            dm=km(open=[col('open').symlog()], close=[col('close').symlog()])
        ),
        dict(
            fn=tulip.rmi,
            pg=km(period=window, lookback_period=lookback_period),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.roc,
            pg=km(period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.rocr,
            pg=km(period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.rsi,
            pg=km(period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.rvi,
            pg=km(ema_period=ema_period, stddev_period=stddev_period),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.smi,
            pg=km(q_period=window, r_period=window, s_period=window),
            dm=km(high=['high'], low=['low'], close=['close'])
        ),
        dict(
            fn=tulip.stddev,
            pg=km(period=window),
            dm=km(series=[col(c).symlog() for c in ('open', 'high', 'low', 'close', 'volume')])
        ),
        dict(
            fn=tulip.stderr,
            pg=km(period=window),
            dm=km(series=[col(c).symlog() for c in ('open', 'high', 'low', 'close', 'volume')])
        ),
        dict(
            fn=tulip.stoch_d,
            pg=km(k_period=[10], k_slowing_period=[20], d_period=[12]),
            dm=km(high=['high'], low=['low'], close=['close'])
        ),
        dict(
            fn=tulip.stoch_k,
            pg=km(k_period=[10], k_slowing_period=[20], d_period=[12]),
            dm=km(high=['high'], low=['low'], close=['close'])
        ),
        dict(
            fn=tulip.stochrsi,
            pg=km(period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.trix,
            pg=km(period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.tsi,
            pg=km(y_period=window, z_period=window),
            dm=km(series=['open', 'high', 'low', 'close', 'volume'])
        ),
        dict(
            fn=tulip.ultosc,
            pg=km(short_period=short_period, medium_period=window, long_period=long_period),
            dm=km(high=['high'], low=['low'], close=['close']),
            cn='p.short_period < p.medium_period < p.long_period'
        ),
        dict(
            fn=tulip.vosc,
            pg=km(short_period=short_period, long_period=long_period),
            dm=km(volume=['volume']),
            cn='p.short_period < p.long_period'
        ),
        dict(
            fn=tulip.willr,
            pg=km(period=window),
            dm=km(high=['high'], low=['low'], close=['close'])
        ),
    ]
)
