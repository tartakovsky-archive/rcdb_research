from importlib import resources

import pytest
import pandas as pd
from commons.utils import get_df_from_hdf_bytes
from commons.features.datetime import components, holidays, markets


@pytest.fixture(scope="module")
def data():
    with resources.open_binary("tests.datasets", "bitfinex__BTC_USD.hdf") as f:
        df = get_df_from_hdf_bytes(f.read())
    return df.index


@pytest.mark.parametrize(
    "calc_all, params_set",
    [
        (components.calc_all, None),
        (holidays.calc_all, [{"country_name": "US"}, {"country_name": "GB"}]),
        (markets.calc_all, [{"market": "NYSE"}])
    ]
)
def test_calc_all(calc_all, params_set, data):
    assert len(data) == len(calc_all(data, params_set))


@pytest.mark.parametrize(
    "calc_all, params_set",
    [
        (components.calc_all, None),
        (holidays.calc_all, [{"country_name": "US"}, {"country_name": "GB"}]),
        (markets.calc_all, [{"market": "NYSE"}])
    ]
)
def test_calc_all(calc_all, params_set, data):
    wrong_data = pd.DataFrame(dict(d=data))

    with pytest.raises(ValueError):
        calc_all(wrong_data, params_set)


@pytest.mark.parametrize(
    "wrong_country_name",
    ["UK", "sada"]
)
def test_holiday_calc_all_params_validation(wrong_country_name, data):
    with pytest.raises(ValueError):
        holidays.calc_all(data, [{"country_name": wrong_country_name}])


@pytest.mark.parametrize(
    "wrong_market_name",
    ["wrong_name", "BTC"]
)
def test_markets_calc_all_params_validation(wrong_market_name, data):
    with pytest.raises(ValueError):
        markets.calc_all(data, [{"market": wrong_market_name}])
