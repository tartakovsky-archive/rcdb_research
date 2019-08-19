import numpy as np
import pandas as pd

from commons.features import direction


TEST_ARR = np.array([1, 2, 5, 3, 4, 1, 0.5])
TEST_DF = pd.DataFrame({"close": TEST_ARR})
WINDOW_SIZE = 3
TEST_RES = [2, 1, 1, -1, -1]

PARAM_SET = {
    "sum_of_direction": [{"column": "close"}]
}


def test_calc_all():
    res = direction.calc_all(TEST_DF, PARAM_SET, WINDOW_SIZE)
    assert not res[f"{direction.PREFIX}_sum_of_direction_3_close"].isnull().all()


def test_sum_of_direction():
    res = direction.sum_of_direction(TEST_ARR, WINDOW_SIZE)
    assert np.array_equal(res, TEST_RES)
