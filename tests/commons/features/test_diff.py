import pytest
import numpy as np

from commons.features import diff

TEST_SERIES = np.array([1., 2., 12., 5., 5.5, 2.])


@pytest.mark.parametrize(
    'func, params, nan_offset',
    [
        (diff.diff, (TEST_SERIES, 5), 5),
        (diff.frac_diff, (TEST_SERIES, 5), 5),
        (diff.series_ma_frac_diff, (TEST_SERIES, 5), 4),
        (diff.two_series_ma_frac_diff, (TEST_SERIES, TEST_SERIES * 2, 5), 4),
    ]
)
def test_diffs(func, params, nan_offset):
    res = func(*params)

    assert np.isnan(res[:nan_offset]).all()
    assert np.isfinite(res[nan_offset:]).all()
