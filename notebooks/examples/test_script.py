from rcdb_research import bars


if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(level=logging.DEBUG)
    from rcdb_research.rcdb_data import RcdbData

    df = RcdbData.fetch("BTC", "UST", "bitfinex",
                        ohlcv_api_url="https://europe-west1-rcdb-prod.cloudfunctions.net/kaiko")

    df = df[df.index >= "2019-04-01"]

    now = time.time()
    df_bars = bars.range.fixed(
        ohlc=df,
        threshold=0.1,
    )
    print(time.time() - now)
    print(df_bars)
