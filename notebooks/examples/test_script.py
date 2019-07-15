# flake8: noqa  
from commons import bars


if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(level=logging.DEBUG)
    from commons.rcdb_data import RcdbData

    df = RcdbData.fetch("BTC", "UST", "bitfinex",
                        ohlcv_api_url="https://europe-west1-rcdb-prod.cloudfunctions.net/kaiko")

    df = df[df.index >= "2019-04-01"]

    # now = time.time()
    # df_bars = bars.range.fixed(
    #     ohlc=df,
    #     threshold=0.1,
    # )
    # print(time.time() - now)
    # print(df_bars)

    DATA_MAPPING = dict(
        series='close',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
    )

    from commons.features import tulip
    features = tulip.adx.calc_all(df, DATA_MAPPING, [{'period': 10}])
