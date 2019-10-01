import numpy as np

from commons.features import diff


TEST_SERIES = np.array([1., 2., 12., 5., 5.5, 2.])


def test_diff():
    res = diff.diff(TEST_SERIES, 5)

    assert np.isnan(res[:5]).all()
    assert np.isfinite(res[5:]).all()


def test_frac_diff():
    res = diff.frac_diff(TEST_SERIES, 5)

    assert np.isnan(res[:5]).all()
    assert np.isfinite(res[5:]).all()


def test_series_ma_frac_diff():
    res = diff.series_ma_frac_diff(TEST_SERIES, 5)

    assert np.isnan(res[:4]).all()
    assert np.isfinite(res[4:]).all()


def test_two_series_ma_frac_diff():
    res = diff.two_series_ma_frac_diff(TEST_SERIES, TEST_SERIES * 2, 5)

    assert np.isnan(res[:4]).all()
    assert np.isfinite(res[4:]).all()
