import pytest
import pandas as pd

from commons.features.stats import calc_all, FEATURE_FUNCS, PREFIX

WINDOW = 5
TEST_LIST = [
    0.8, 4.33, 4.83, 2.31, 3.13,
    4.87, 1.9, 4.84, 2.05, 3.37,
    2.37, 2.16, 1.43, 0.63, 1.26,
    1.01, 0.57, 3.57, 4.91, 0.64
]
MULTIPLE_COLUMN_DF = pd.DataFrame({"val1": TEST_LIST, "val2": TEST_LIST})


PARAMS_SET = {k: [{}] for k in FEATURE_FUNCS}


def test_calc_all():
    res = calc_all(MULTIPLE_COLUMN_DF, PARAMS_SET, WINDOW)
    assert len(res) == len(MULTIPLE_COLUMN_DF) - WINDOW + 1
    assert {f"{PREFIX}_{f}_{WINDOW}_{c}" for f in FEATURE_FUNCS for c in MULTIPLE_COLUMN_DF.columns} == set(res.columns)


@pytest.mark.parametrize(
    "feature",
    list(FEATURE_FUNCS.values()),
    ids=[*FEATURE_FUNCS.keys()]
)
def test_features(feature):
    expected_size = len(MULTIPLE_COLUMN_DF) - WINDOW + 1
    assert feature(MULTIPLE_COLUMN_DF.val1.values, WINDOW).size == expected_size
