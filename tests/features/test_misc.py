import pytest
import numpy as np

from rcdb_research.features import misc

TEST_SERIES = np.array([1., 2., 12., 5., 5.5, 2.])


@pytest.mark.parametrize(
    'func, params, nan_offset, test_res',
    [
        (
            misc.diff,
            (TEST_SERIES, 1),
            1,
            np.array([np.nan, 1.0, 10.0, -7.0, 0.5, -3.5])
        ),
        (
            misc.frac_change,
            (TEST_SERIES, 2),
            2,
            np.array([np.nan, np.nan, 11.0, 1.5, -0.5416666666666666, -0.6])
        ),
        (
            misc.series_ma_frac_change,
            (TEST_SERIES, 3),
            2,
            np.array([np.nan, np.nan, 1.4, -0.21052631578947367, -0.2666666666666667, -0.5199999999999999])
        ),
        (
            misc.two_series_ma_frac_change,
            (TEST_SERIES, TEST_SERIES * 2, 5),
            4,
            np.array([np.nan, np.nan, np.nan, np.nan, -0.5, -0.5])
        ),
        (
            misc.direction,
            (np.array([1, 2, 3, 2]), np.array([3, 2, 1, 4])),
            0,
            np.array([1, 0, 0, 1])
        ),
        (
            misc.frac_change_open_to_close,
            (np.array([1, 2, 2, 2]), np.array([3, 2, 1, 4])),
            0,
            np.array([2.0, 0.0, -0.5, 1.0])
        ),
        (
            misc.exposure,
            (np.array([1, 2, 2, 2]), np.array([3, 2, 1, 4])),
            0,
            np.array([-2.0, 0.0, 1, -2])
        ),
    ]
)
def test_diffs(func, params, nan_offset, test_res):
    res = func(*params)

    assert np.isnan(res[:nan_offset]).all()
    assert np.isfinite(res[nan_offset:]).all()
    assert np.array_equal(
        res[nan_offset:],
        test_res[nan_offset:]
    )
