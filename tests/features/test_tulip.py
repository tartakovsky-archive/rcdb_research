import inspect

import pytest
import numpy as np

from rcdb_research.features import tulip


@pytest.fixture
def df(ohlcv_df):
    data = ohlcv_df
    data['volume'] = data.volume_buy + data.volume_sell
    yield data


@pytest.mark.parametrize(
    "feature, params, nan_end",
    [
        (tulip.adosc, {'short_period': 2, 'long_period': 5}, 4),
        (tulip.adx, {'period': 10}, 18),
        (tulip.adxr, {'period': 10}, 27),
        (tulip.ao, {}, 33),
        (tulip.apo, {'short_period': 2, 'long_period': 5}, 1),
        (tulip.aroonosc, {'period': 10}, 10),
        (tulip.cci, {'period': 10}, 18),
        (tulip.cmf, {'period': 10}, 9),
        (tulip.cmo, {'period': 10}, 10),
        (tulip.copp, {'roc_shorter_period': 2, 'roc_longer_period': 10, 'wma_period': 5}, 14),
        (tulip.cvi, {'period': 10}, 19),
        (tulip.dpo, {'period': 10}, 9),
        (tulip.dx, {'period': 10}, 9),
        (tulip.emv, {}, 1),
        (tulip.fisher, {'period': 10}, 9),
        (tulip.fisher_signal, {'period': 10}, 9),
        (tulip.fosc, {'period': 10}, 10),
        (tulip.linregslope, {'period': 10}, 9),
        (tulip.macd, {'short_period': 3, 'long_period': 6, 'signal_period': 5}, 5),
        (tulip.macd_signal, {'short_period': 3, 'long_period': 6, 'signal_period': 5}, 5),
        (tulip.marketfi, {}, 0),
        (tulip.mass, {'period': 10}, 25),
        (tulip.md, {'period': 10}, 9),
        (tulip.mfi, {'period': 10}, 10),
        (tulip.natr, {'period': 10}, 9),
        (tulip.pfe, {'period': 10, 'ema_period': 11}, 10),
        (tulip.posc, {'period': 10, 'ema_period': 11}, 9),
        (tulip.ppo, {'short_period': 2, 'long_period': 5}, 1),
        (tulip.qstick, {'period': 10}, 9),
        (tulip.rmi, {'period': 10, 'lookback_period': 11}, 11),
        (tulip.roc, {'period': 10}, 10),
        (tulip.rocr, {'period': 10}, 10),
        (tulip.rsi, {'period': 10}, 10),
        (tulip.rvi, {'ema_period': 10, 'stddev_period': 11}, 10),
        (tulip.smi, {'q_period': 10, 'r_period': 10, 's_period': 10}, 9),
        (tulip.stddev, {'period': 10}, 9),
        (tulip.stderr, {'period': 10}, 9),
        (tulip.stochrsi, {'period': 10}, 19),
        (tulip.trix, {'period': 10}, 28),
        (tulip.tsi, {'y_period': 10, 'z_period': 10}, 1),
        (tulip.ultosc, {'short_period': 2, 'medium_period': 4, 'long_period': 5}, 5),
        (tulip.vosc, {'short_period': 2, 'long_period': 5}, 4),
        (tulip.willr, {'period': 10}, 9),
        (tulip.kst_signal, {
            'roc1': 10, 'roc2': 15, 'roc3': 20, 'roc4': 30, 'ma1': 10, 'ma2': 10, 'ma3': 10, 'ma4': 15
        }, 30),
        (tulip.minus_di, {'period': 10}, 9),
        (tulip.hurst, {'period': 10}, 9),
        (tulip.minus_dm, {'period': 10}, 9),
        (tulip.plus_di, {'period': 10}, 9),
        (tulip.plus_dm, {'period': 10}, 9),
        (tulip.msw_sine, {'period': 10}, 10),
        (tulip.stoch_k, {'k_period': 10, 'k_slowing_period': 20, 'd_period': 12}, 39),
        (tulip.stoch_d, {'k_period': 10, 'k_slowing_period': 20, 'd_period': 12}, 39),
    ]
)
def test_tulip(df, feature, params, nan_end):
    cols = set(df.columns.values)
    if "series" in inspect.getfullargspec(feature).args:
        inputs = {'series': df.close.values}
    else:
        inputs = {k: df[k].values for k in inspect.getfullargspec(feature).args if k in cols}

    res = feature(**{**inputs, **params})
    assert len(res) == len(df)
    if nan_end:
        assert np.isnan(res[:nan_end]).all()
        assert np.isreal(res[nan_end:]).all()
