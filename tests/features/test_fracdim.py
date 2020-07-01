import pytest
import numpy as np

from rcdb_research.features import fracdim

WINDOW = 20
TEST_SERIES = np.array(
    [
        0.8, 4.33, 4.83, 2.31, 3.13,
        4.87, 1.9, 4.84, 2.05, 3.37,
        2.37, 2.16, 1.43, 0.63, 1.26,
        1.01, 0.57, 3.57, 4.91, 0.64
    ] * 30
)


@pytest.mark.parametrize(
    "feature",
    [fracdim.petrosian_fd, fracdim.katz_fd, fracdim.higuchi_fd]
)
def test_features(feature):
    res = feature(TEST_SERIES, WINDOW)
    assert res.size == TEST_SERIES.size
    assert np.isfinite(res).any()
