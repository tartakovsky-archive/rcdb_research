import pandas as pd

from commons.utils import store_df_to_hdf_bytes, get_df_from_hdf_bytes


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    df = pd.DataFrame([1, 2, 3])
    print(df)

    b = store_df_to_hdf_bytes(df)
    print(get_df_from_hdf_bytes(b))

    from commons.ohlcv import OHLCV

    df = OHLCV.fetch("BTC", "USD", "bitfinex", "1m", ohlcv_api_url="test-back")
    print(df)
