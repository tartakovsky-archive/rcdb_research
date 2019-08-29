import pytest
import time

from importlib import resources
from commons.utils import get_df_from_hdf_bytes
from commons.features.datetime import holidays


@pytest.fixture(scope="module")
def data():
    with resources.open_binary("tests.datasets", "bitfinex__BTC_USD.hdf") as f:
        df = get_df_from_hdf_bytes(f.read())
    return df[["high", "low", "close"]]


@pytest.mark.skip
class TestParallelCalcAll:
    def test_parallel(self, data):
        start = time.time()

        # n_jobs == 1 - no parallel computing code is used at all
        df1 = holidays.calc_all(data.index, [dict(country_name="US") for i in range(0, 10)], n_jobs=1)

        # n_jobs == -1 - parallel to all available CPUs
        df2 = holidays.calc_all(data.index, [dict(country_name="US") for i in range(0, 10)], n_jobs=-1)
        assert df1.equals(df2)

        print(time.time() - start)
