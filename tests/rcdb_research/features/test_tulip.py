import os

import pandas as pd
import pytest

from rcdb_research.features import tulip

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
def df():
    data = pd.read_hdf(DATASET, key='table')
    data['volume'] = data.volume_buy + data.volume_sell
    yield data


def test_source_data_does_not_mutate(df):
    copy = df.copy()
    tulip.adx.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'period': 10
        }],
        feature_param_sets=[{
            'n': 1
        }],
        inplace=False,
    )
    assert df.equals(copy)


def test_tulip_namespaces():
    assert 'bbands' in tulip.namespaces


def test_global_calc_all(df):
    indicators_param_sets = {
        'stoch': [{
            'k_period': 10,
            'k_slowing_period': 20,
            'd_period': 12,
        }],
        'cci': [{
            'period': 10,
        }],
        'adx': [{
            'period': 10,
        }],
        'macd': [{
            'short_period': 3,
            'long_period': 6,
            'signal_period': 5,
        }],
        'bbands': [{
            'period': 10,
            'stddev': 2,
        }],
        'willr': [{
            'period': 10,
        }],
        'rsi': [{
            'period': 10,
        }],
        'roc': [{
            'period': 10,
        }],
        'psar': [{
            'acceleration_factor_step': .2,
            'acceleration_factor_maximum': 2,
        }],
    }
    features_param_sets = {
        'obv': [{
            'n': 1
        }],
        'bop': [{
            'n': 1
        }],
        'stoch': [{
            'n': 1
        }],
        'cci': [{
            'n': 1
        }],
        'adx': [{
            'n': 1
        }],
        'macd': [{
            'n': 1
        }],
        'bbands': [{
            'n': 1
        }],
        'willr': [{
            'n': 1
        }],
        'rsi': [{
            'n': 1
        }],
        'roc': [{
            'n': 1
        }],
        'psar': [{
            'n': 1
        }],
    }
    features = tulip.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicators_param_sets=indicators_param_sets,
        features_param_sets=features_param_sets,
    )
    split_names = [c.split('_') for c in features.columns]
    assert all(n[0] == 'tulip' for n in split_names)
    assert all(n[1] in tulip.namespaces for n in split_names)


def test_global_inputs():
    assert is_match_up(tulip.inputs,
                       ('series', 'open', 'high', 'low', 'close', 'volume'))


def test_adx(df):
    features = tulip.adx.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'period': 10
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.adx.inputs, ('low', 'high'))
    assert not features.empty


def test_bbands(df):
    features = tulip.bbands.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'period': 10,
            'stddev': 2
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.bbands.inputs, ('close', 'series'))
    assert not features.empty


def test_bop(df):
    features = tulip.bop.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.bop.inputs, ('low', 'close', 'open', 'high'))
    assert not features.empty


def test_cci(df):
    features = tulip.cci.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'period': 10
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.cci.inputs, ('high', 'close', 'low'))
    assert not features.empty


def test_macd(df):
    features = tulip.macd.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'short_period': 3,
            'long_period': 6,
            'signal_period': 5
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.macd.inputs, ('series', ))
    assert not features.empty


def test_obv(df):
    features = tulip.obv.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.obv.inputs, ('volume', 'series'))
    assert not features.empty


def test_psar(df):
    features = tulip.psar.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'acceleration_factor_step': .2,
            'acceleration_factor_maximum': 2
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.psar.inputs, ('high', 'low', 'series'))
    assert not features.empty


def test_roc(df):
    features = tulip.roc.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'period': 10
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.roc.inputs, ('series', ))
    assert not features.empty


def test_rsi(df):
    features = tulip.rsi.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'period': 10
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.rsi.inputs, ('series', ))
    assert not features.empty


def test_stoch(df):
    features = tulip.stoch.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'k_period': 10,
            'k_slowing_period': 20,
            'd_period': 12
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.stoch.inputs, ('high', 'low', 'close'))
    assert not features.empty


def test_willr(df):
    features = tulip.willr.calc_all(
        data=df,
        data_mapping=DATA_MAPPING,
        indicator_param_sets=[{
            'period': 10
        }],
        feature_param_sets=[{
            'n': 1
        }],
    )
    assert is_match_up(tulip.willr.inputs, ('high', 'low', 'close'))
    assert not features.empty
