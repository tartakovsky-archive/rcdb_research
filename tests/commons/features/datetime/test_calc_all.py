import pytest
import pandas as pd
from commons.features.datetime import components, holidays, markets, calc_all


DT_CALC_ALL_PARAMS_SETS = {
    holidays.PREFIX: [{"country_name": "US"}],
    markets.PREFIX: [{"market": "NYSE"}]
}


@pytest.fixture(scope="module")
def data(ohlcv_df):
    return ohlcv_df.index


@pytest.mark.parametrize(
    "calc_all, params_set",
    [
        (components.calc_all, None),
        (holidays.calc_all, [{"country_name": "US"}, {"country_name": "GB"}]),
        (markets.calc_all, [{"market": "NYSE"}]),
        (calc_all, DT_CALC_ALL_PARAMS_SETS)
    ],
    ids=["calc_all_components", "calc_all_holidays", "calc_all_markets", "calc_all_dt"]
)
def test_calc_all(calc_all, params_set, data):
    assert len(data) == len(calc_all(data, params_set))

    with pytest.raises(ValueError):
        wrong_data = pd.DataFrame(dict(d=data))
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


def test_imports(data):
    dt_index = data[:100]

    import commons

    commons.features.datetime.calc_all(dt_index, DT_CALC_ALL_PARAMS_SETS)
    commons.features.datetime.holidays.calc_all(dt_index, [{"country_name": "US"}])
    commons.features.datetime.holidays.f6(dt_index.to_pydatetime())

    # -----

    from commons import features

    features.datetime.calc_all(dt_index, DT_CALC_ALL_PARAMS_SETS)

    # -----

    from commons.features import datetime

    datetime.calc_all(dt_index, DT_CALC_ALL_PARAMS_SETS)

    # -----

    from commons.features.datetime import components

    components.calc_all(dt_index)
