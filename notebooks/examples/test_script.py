from commons import bars


if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(level=logging.DEBUG)
    from commons.ohlcv import OHLCV

    df = OHLCV.fetch("BTC", "UST", "bitfinex",
                     ohlcv_api_url="https://europe-west1-rcdb-prod.cloudfunctions.net/kaiko")

    df = df[df.index >= "2019-04-01"]

    now = time.time()
    df_bars = bars.hybrid.range_fixed_volume_adaptive(
        ohlc=df,
        pct_threshold=0.1,
        avg_per=3,
        window=6
    )
    print(time.time() - now)
    print(df_bars)
