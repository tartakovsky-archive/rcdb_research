import pytest
import pandas as pd

from commons.features.mne import calc_all, FEATURE_FUNCS, PREFIX

TEST_LIST = [
    0.8, 4.33, 4.83, 2.31, 3.13,
    4.87, 1.9, 4.84, 2.05, 3.37,
    2.37, 2.16, 1.43, 0.63, 1.26,
    1.01, 0.57, 3.57, 4.91, 0.64
]
MULTIPLE_COLUMN_DF = pd.DataFrame({"val1": TEST_LIST, "val2": TEST_LIST})
SINGLE_COLUMN_DF = pd.DataFrame(TEST_LIST * 20, columns=["val"])

@pytest.mark.parametrize(
    "data,param_set,window,column_names",
    [
        (
            pd.DataFrame(TEST_LIST, columns=["val"]),
            dict(f30=[dict(column="val")]),
            18,
            None
        ),
        *[
            (
                SINGLE_COLUMN_DF,
                {feature: [dict(column="val")]},
                350,
                dict(val="val")
            )
            for feature in FEATURE_FUNCS if feature != "f30"
        ]
    ],
    ids=["f30"] + [feature for feature in FEATURE_FUNCS if feature != "f30"]
)
def test_calc_all(data, param_set, window, column_names):
    res = calc_all(data, param_set, window, column_names)
    assert len(res) == len(data) - window + 1


def test_calc_all_column_param():
    ps = {
        "f1": [{"column": "val2"}]
    }
    res = calc_all(MULTIPLE_COLUMN_DF, ps, 3)
    assert {f"{PREFIX}_f1_3_val2"} == set(res.columns)


@pytest.mark.parametrize(
    "ps",
    [
        {"f1": [{"column": "val2"}, {"column": "val1"}]},
        {"f1": [{"column": "__all__"}]},
        {"f1": [{}]},
    ]
)
def test_calc_all_column_param__all__(ps):
    res = calc_all(MULTIPLE_COLUMN_DF, ps, 3)
    assert {f"{PREFIX}_f1_3_val1", f"{PREFIX}_f1_3_val2"} == set(res.columns)


@pytest.mark.parametrize(
    "ps",
    [
        {"f8": [{"emb": 3}, {"emb": 4}]},
        {"f10": [{"sfreq": 270.}]},
    ]
)
def test_calc_all_custom_feature_params(ps):
    res = calc_all(SINGLE_COLUMN_DF, ps, 300)
    assert len(res.columns) == len(list(ps.values())[0])
