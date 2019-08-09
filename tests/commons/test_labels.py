import pytest
import numpy as np

from commons.labeling import higher_after_n_bars, lower_after_n_bars, \
    n_consecutive_up, n_consecutive_down


TEST_SERIES = np.array([1, 2, 3, 4, 2, 1, 3, 5])
TEST_N = 2


@pytest.mark.parametrize(
    "func, test_not_nan_part",
    [
        (higher_after_n_bars, [1, 0, 0, 0, 1]),
        (lower_after_n_bars, [0, 0, 1, 1, 0]),
        (n_consecutive_up, [1, 1, 0, 0, 0, 1]),
        (n_consecutive_down, [0, 0, 0, 1, 0, 0]),
    ]
)
def test_labeling(func, test_not_nan_part):
    r = func(TEST_SERIES, TEST_N)

    nan_part_size = r.size - len(test_not_nan_part)
    assert np.isnan(r[-nan_part_size:]).all()

    assert np.array_equal(
        r[:-nan_part_size],
        test_not_nan_part
    )
