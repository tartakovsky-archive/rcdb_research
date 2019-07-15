import os
import pytest
import pandas as pd

from commons.features import tulip

ERROR = 1e-9
DATA_MAPPING = dict(
    series='close',
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
)
DATASET = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../datasets/bitfinex__BTC_USD.hdf"))
PREFIX = 'tulip'


def is_match_up(seq1, seq2):
    return len(set(seq1) & set(seq2)) == len(seq1) == len(seq2)


@pytest.fixture
def test_dataset():
    df = pd.read_hdf(DATASET, key='table')
    df['volume'] = df.volume_buy + df.volume_sell
    yield df


def test_source_data_does_not_mutate(test_dataset):
    source = test_dataset
    copy = source.copy()
    tulip.adx.calc_all(
        test_dataset, DATA_MAPPING, [{'period': 10}], inplace=False)
    assert source.equals(copy)


def test_tulip_namespaces():
    assert len(tulip.namespaces) == 11


# def test_global_calc_all(test_dataset):
#     tulip


def test_adx(test_dataset):
    features = tulip.adx.calc_all(test_dataset, DATA_MAPPING, [{'period': 10}])
    assert is_match_up(tulip.adx.inputs, ('low', 'high'))
    assert [f"{PREFIX}_adx_{f.__name__}_10" for f in tulip.adx.features_list] \
        in features.columns.values


def test_bbands(test_dataset):
    features = tulip.bbands.calc_all(
        test_dataset, DATA_MAPPING, [{'period': 10, 'stddev': 2}])
    assert is_match_up(tulip.bbands.inputs, ('low', 'close', 'open', 'high'))
    assert [f"{PREFIX}_bbands_{f.__name__}_10_2"
            for f in tulip.bbands.features_list] \
        in features.columns.values


def test_bop(test_dataset):
    features = tulip.bop.calc_all(test_dataset, DATA_MAPPING)
    assert is_match_up(tulip.bop.inputs, ('low', 'close', 'open', 'high'))
    assert [f"{PREFIX}_bop_{f.__name__}" for f in tulip.bop.features_list] \
        in features.columns.values


def test_cci(test_dataset):
    features = tulip.cci.calc_all(test_dataset, DATA_MAPPING, [{'period': 10}])
    assert is_match_up(tulip.cci.inputs, ('high', 'close', 'low'))
    assert [f"{PREFIX}_cci_{f.__name__}_10" for f in tulip.cci.features_list] \
        in features.columns.values


def test_macd(test_dataset):
    features = tulip.macd.calc_all(
        test_dataset,
        DATA_MAPPING,
        [{'short_period': 3, 'long_period': 6, 'signal_period': 5}])
    assert is_match_up(tulip.macd.inputs, ('series', ))
    assert [f"{PREFIX}_macd_{f.__name__}_3_6_5"
            for f in tulip.macd.features_list] \
        in features.columns.values


def test_obv(test_dataset):
    features = tulip.obv.calc_all(test_dataset, DATA_MAPPING)
    assert is_match_up(tulip.obv.inputs, ('volume', 'series'))
    assert [f"{PREFIX}_obv_{f.__name__}" for f in tulip.obv.features_list] \
        in features.columns.values


def test_psar(test_dataset):
    features = tulip.psar.calc_all(
        test_dataset,
        DATA_MAPPING,
        [{'acceleration_factor_step': .2, 'acceleration_factor_maximum': 2}])
    assert is_match_up(tulip.psar.inputs, ('open', 'high', 'low', 'close'))
    assert [f"{PREFIX}_psar_{f.__name__}_0.2_2"
            for f in tulip.psar.features_list] \
        in features.columns.values


def test_roc(test_dataset):
    features = tulip.roc.calc_all(test_dataset, DATA_MAPPING, [{'period': 10}])
    assert is_match_up(tulip.roc.inputs, ('series', ))
    assert [f"{PREFIX}_roc_{f.__name__}_10" for f in tulip.roc.features_list] \
        in features.columns.values


def test_rsi(test_dataset):
    features = tulip.rsi.calc_all(test_dataset, DATA_MAPPING, [{'period': 10}])
    assert is_match_up(tulip.rsi.inputs, ('series', ))
    assert [f"{PREFIX}_rsi_{f.__name__}_10" for f in tulip.rsi.features_list] \
        in features.columns.values


def test_stoch(test_dataset):
    features = tulip.stoch.calc_all(
        test_dataset,
        DATA_MAPPING,
        [{'k_period': 10, 'k_slowing_period': 20, 'd_period': 12}])
    assert is_match_up(tulip.stoch.inputs, ('high', 'low', 'close'))
    assert [f"{PREFIX}_stoch_{f.__name__}_10_20_12"
            for f in tulip.stoch.features_list] \
        in features.columns.values


def test_willr(test_dataset):
    features = tulip.willr.calc_all(
        test_dataset, DATA_MAPPING, [{'period': 10}])
    assert is_match_up(tulip.willr.inputs, ('high', 'low', 'close'))
    assert [f"{PREFIX}_willr_{f.__name__}_10"
            for f in tulip.willr.features_list] \
        in features.columns.values
