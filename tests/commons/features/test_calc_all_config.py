import pytest
import time

from importlib import resources
from commons.utils import get_df_from_hdf_bytes
from commons.utils import calc_all_config, km, t
from commons import features


@pytest.fixture(scope="module")
def data():
    with resources.open_binary("tests.datasets", "bitfinex__BTC_USD.hdf") as f:
        df = get_df_from_hdf_bytes(f.read())
    return df[["high", "low", "close"]]


@pytest.mark.skip
class TestSettings:
    def test_parallel(self, data):
        print("-")
        start = time.time()
        print(time.time() - start)
        config = dict(
            prefix_1=[
                dict(
                    fn=features.datetime.holidays.f1,
                    pg=km(country_name=['US', 'RU']),
                    dm=km(timestamps=[km.col("index").t([t.custom_transform(1, "OK")])]),
                    tr=[t.symlog()]
                )
            ]
        )
        df1 = calc_all_config(data, config=config)  # n_jobs==1 by default
        df2 = calc_all_config(data, config=config, n_jobs=-1)
        assert df1.equals(df2)
