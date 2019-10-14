import numpy as np
import pandas as pd

from rcdb_research.features import direction


TEST_ARR = np.array([1, 2, 5, 3, 4, 1, 0.5])
TEST_DF = pd.DataFrame({"close": TEST_ARR})
WINDOW_SIZE = 3
TEST_RES = [np.nan, np.nan, 2, 1, 1, -1, -1]

PARAM_SET = {
    "sum_of_direction": [{"column": "close"}]
}


def test_calc_all():
    res = direction.calc_all(TEST_DF, PARAM_SET, WINDOW_SIZE)
    assert len(res) == len(TEST_DF)
    assert hasattr(res, f"{direction.PREFIX}_sum_of_direction_3_close")


def test_sum_of_direction():
    res = direction.sum_of_direction(TEST_ARR, WINDOW_SIZE)

    assert np.isnan(res[:WINDOW_SIZE - 1]).all()
    assert np.array_equal(res[WINDOW_SIZE - 1:], TEST_RES[WINDOW_SIZE - 1::])
