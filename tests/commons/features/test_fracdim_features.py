import pytest
import pandas as pd

from commons.features.fracdim import calc_all, FEATURE_FUNCS, PREFIX

WINDOW = 20
TEST_LIST = [
    0.8, 4.33, 4.83, 2.31, 3.13,
    4.87, 1.9, 4.84, 2.05, 3.37,
    2.37, 2.16, 1.43, 0.63, 1.26,
    1.01, 0.57, 3.57, 4.91, 0.64
] * 30
MULTIPLE_COLUMN_DF = pd.DataFrame({"val1": TEST_LIST, "val2": TEST_LIST})


PARAMS_SET = {k: [{}] for k in FEATURE_FUNCS}
PARAMS_SET["higuchi_fd"][0]["kmax"] = 2


def test_calc_all():
    res = calc_all(MULTIPLE_COLUMN_DF, PARAMS_SET, WINDOW)
    assert len(res) == len(MULTIPLE_COLUMN_DF) - WINDOW + 1

    cnames = {
        f"{PREFIX}_{f}_{WINDOW}_{c}{'_' if p else ''}{'_'.join(f'{k}{v}' for k, v in p.items())}"
        for f in FEATURE_FUNCS for c in MULTIPLE_COLUMN_DF.columns for p in PARAMS_SET[f]
    }
    assert cnames == set(res.columns)


@pytest.mark.parametrize(
    "feature, params",
    [
        (FEATURE_FUNCS[fname], PARAMS_SET[fname][0])
        for fname in FEATURE_FUNCS
    ],
    ids=[*FEATURE_FUNCS.keys()]
)
def test_features(feature, params):
    expected_size = len(MULTIPLE_COLUMN_DF) - WINDOW + 1
    assert feature(MULTIPLE_COLUMN_DF.val1.values, WINDOW, **params).size == expected_size
