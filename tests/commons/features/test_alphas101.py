from importlib import resources

import pytest

from commons.features import alphas101
from commons.utils import get_df_from_hdf_bytes

@pytest.fixture(scope="module")
def data():
    with resources.open_binary("tests.datasets", "bitfinex__BTC_USD.hdf") as f:
        df = get_df_from_hdf_bytes(f.read())
    return df


def test_calc_all(data):
    res = alphas101.calc_all(data)
    assert len(res) == len(data)
    assert {f"{alphas101.PREFIX}_{x}" for x in alphas101.FEATURE_FUNCS} == set(res.columns)
