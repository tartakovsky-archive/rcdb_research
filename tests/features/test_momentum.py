import pytest
import numpy as np

from rcdb_research.features import momentum


@pytest.fixture
def price(ohlcv_df):
    return ohlcv_df.close.values


@pytest.fixture()
def volume(ohlcv_df):
    return ohlcv_df.volume_sell.values + ohlcv_df.volume_buy.values


def test_distance(price):
    v = momentum.distance(
        price, min_window=10, max_window=100, number_of_rollings=10
    )
    assert v.shape == price.shape
    assert np.isnan(v[:99]).all()
    assert np.logical_or(v < 1, v > 0)[99:].all()


def test_correlation(price):
    v = momentum.correlation(
        price, min_window=10, max_window=100, number_of_rollings=10
    )
    assert v.shape == price.shape
    assert np.isnan(v[:99]).all()
    assert np.logical_or(v < 1, v > -1)[99:].all()


def test_thickness(price):
    v = momentum.thickness(price, min_window=10, max_window=100, number_of_rollings=10)
    assert v.shape == price.shape
    assert np.isnan(v[:99]).all()
    assert (~np.isnan(v[99:])).all()


@pytest.mark.parametrize('func', [momentum.p0, momentum.p1, momentum.p2, momentum.p3])
def test_p(func, price, volume):
    k = 30
    if func in [momentum.p1, momentum.p2]:
        v = func(price=price, volume=volume, k=k)
    else:
        v = func(price=price, k=k)

    assert price.shape == v.shape
    assert np.isnan(v[:k]).all()
    assert (~np.isnan(v[k:])).all()
