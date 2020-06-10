import pytest
import numpy as np

from rcdb_research.labeling.legacy import higher_after_n_bars, lower_after_n_bars, \
    n_consecutive_up, n_consecutive_down


TEST_SERIES = np.array([1, 2, 3, 4, 2, 1, 3, 5])
TEST_N = 2


@pytest.mark.parametrize(
    "func, test_res",
    [
        (higher_after_n_bars, [1, 1, 0, 0, 1, 1, np.nan, np.nan]),
        (lower_after_n_bars, [0, 0, 1, 1, 0, 0, np.nan, np.nan]),
        (n_consecutive_up, [1, 1, 0, 0, 0, 1, np.nan, np.nan]),
        (n_consecutive_down, [0, 0, 0, 1, 0, 0, np.nan, np.nan]),
    ]
)
def test_labeling(func, test_res):
    r = func(TEST_SERIES, TEST_N)
    assert np.isnan(r[-TEST_N:]).all()

    assert np.array_equal(
        r[:-TEST_N],
        test_res[:-TEST_N]
    )
