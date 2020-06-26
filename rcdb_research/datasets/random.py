import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import laplace, beta, uniform, kappa3


def generate_btcusd_like_random_df(size=1200000, seed=1, start_date='2014-01-01 00:00:00'):
    high_shadow = kappa3.rvs(2, loc=0, scale=0.0007, size=size, random_state=seed)
    low_shadow = -kappa3.rvs(2, loc=0, scale=0.0007, size=size, random_state=seed + 2)
    close_change = laplace.rvs(loc=0.0, scale=10, size=size, random_state=seed + 4)

    close = np.cumsum(close_change) + 100
    opn = pd.Series(close).shift().fillna(100).values
    high = (1 + high_shadow) * np.where(close > opn, close, opn)
    low = (1 + low_shadow) * np.where(close < opn, close, opn)

    volume = kappa3.rvs(3, loc=0, scale=60, size=size, random_state=seed)
    volume_buy_pct = beta.rvs(0.55, 0.45, loc=0, scale=1, size=size, random_state=seed)
    volume_buy = volume * volume_buy_pct
    volume_sell = volume * (1 - volume_buy_pct)
    volume_quote = volume * close
    volume_quote_buy = volume_buy * close
    volume_quote_sell = volume_sell * close

    ticks = kappa3.rvs(3, loc=0, scale=10, size=size, random_state=seed).astype(int)
    ticks_buy_pct = uniform.rvs(loc=0, scale=1, size=size, random_state=seed + 1)
    ticks_buy = (ticks * ticks_buy_pct).astype(int)
    ticks_sell = ticks - ticks_buy

    dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    ts = dt.timestamp()
    minutes = np.arange(close.shape[0]) * 60
    timestamps = (minutes + ts).astype('datetime64[s]')
    index = pd.DatetimeIndex(timestamps)

    df = pd.DataFrame(
        np.array([opn, high, low, close,
                  volume, volume_buy, volume_sell,
                  volume_quote, volume_quote_buy, volume_quote_sell,
                  ticks, ticks_buy, ticks_sell]).T,
        columns=['open', 'high', 'low', 'close',
                 'volume', 'volume_buy', 'volume_sell',
                 'volume_quote', 'volume_quote_buy', 'volume_quote_sell',
                 'ticks', 'ticks_buy', 'ticks_sell'],
        index=index
    )
    return df

# Usage
# rand_1m = generate_btcusd_like_random_df(size=1200000)
# bars_rnd = consolidators.percent(rand_1m, threshold=bar_size)
# bars_rnd['timestamp'] = bars_rnd.index.values.astype("int64") / 1e9
