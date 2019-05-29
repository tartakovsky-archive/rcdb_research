
import pandas as pd

from commons.utils import store_df_to_hdf_bytes, get_df_from_hdf_bytes
from commons.consolidators import min_pct_bars

if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(level=logging.DEBUG)
    from commons.ohlcv import OHLCV

    df = OHLCV.fetch("BTC", "UST", "bitfinex",
                     ohlcv_api_url="https://europe-west1-rcdb-prod.cloudfunctions.net/kaiko")

    # df = df[df.index >= "2019-04-01"]

    print(df.head())
    print(df.tail())
    print(df.columns)

    # now = time.time()
    # bars = min_pct_bars(df, 0.02)
    #
    # print(time.time() - now)
    # print(bars.close.pct_change())

