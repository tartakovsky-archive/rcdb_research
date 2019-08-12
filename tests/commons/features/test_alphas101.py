from commons.features import alphas101


def test_calc_all(ohlcv_df):
    res = alphas101.calc_all(ohlcv_df)
    assert len(res) == len(ohlcv_df)
    assert {f"{alphas101.PREFIX}_{x}" for x in alphas101.FEATURE_FUNCS} == set(res.columns)
