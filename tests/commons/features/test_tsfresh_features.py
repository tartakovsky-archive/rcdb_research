import pytest
import pandas as pd

from commons.features.tsfresh import calc_all, FEATURE_FUNCS, PREFIX

WINDOW = 5
TEST_LIST = [
    0.8, 4.33, 4.83, 2.31, 3.13,
    4.87, 1.9, 4.84, 2.05, 3.37,
    2.37, 2.16, 1.43, 0.63, 1.26,
    1.01, 0.57, 3.57, 4.91, 0.64
]
MULTIPLE_COLUMN_DF = pd.DataFrame({"val1": TEST_LIST, "val2": TEST_LIST})
PARAMS_SET = {
    "f1": [{}],
    "f2": [{"column": "val2"}],
    "f3": [dict(column="val1", maxlag=3, f_agg="mean")],
    "f4": [dict(column="val1", chunk_len=3, attr="rvalue", f_agg="mean")],
    "f5": [dict(column="val1", m=2, r=.5)],
    "f6": [dict(column="val1", coeff=0.5, k=2)],
    "f7": [dict(column="val1", attr="teststate")],
    "f8": [dict(column="val1", lag=3)],
    "f9": [dict(column="val1", max_bins=2)],
    "f10": [dict(column="val1", lag=3)],
    "f11": [dict(column="val1", ql=.05, qh=.2, isabs=True, f_agg="mean")],
    "f12": [dict(column="val1", normalize=True)],
    "f13": [dict(column="val1", )],
    "f14": [dict(column="val1", )],
    "f15": [dict(column="val1", num_segments=2, segment_focus=1)],
    "f16": [dict(column="val1", )],
    "f17": [dict(column="val1", )],
    "f18": [dict(column="val1", )],
    "f19": [dict(column="val1", )],
    "f20": [dict(column="val1", coeff=0, attr="abs")],
    "f21": [dict(column="val1", )],
    "f22": [dict(column="val1", )],
    # "f23": [dict(column="val1", m=2, r=.7, coeff=1)],
    "f24": [dict(column="val1", )],
    "f25": [dict(column="val1", )],
    "f26": [dict(column="val1", )],
    "f27": [dict(column="val1", q=.3)],
    "f28": [dict(column="val1", )],
    "f29": [dict(column="val1", r=.25)],
    "f30": [dict(column="val1", )],
    "f31": [dict(column="val1", )],
    "f32": [dict(column="val1", attr="pvalue")],
    "f33": [dict(column="val1", )],
    "f34": [dict(column="val1", )],
    # "f35": [dict(column="val1", m=2, r=0.5)],
    "f36": [dict(column="val1", )],
    "f37": [dict(column="val1", )],
    "f38": [dict(column="val1", )],
    "f39": [dict(column="val1", )],
    "f40": [dict(column="val1", )],
    "f41": [dict(column="val1", )],
    "f42": [dict(column="val1", )],
    "f43": [dict(column="val1", m=.1)],
    "f44": [dict(column="val1", n=3)],
    "f45": [dict(column="val1", lag=3)],
    "f46": [dict(column="val1", )],
    "f47": [dict(column="val1", )],
    "f48": [dict(column="val1", q=.5)],
    "f49": [dict(column="val1", min=0.1, max=1.)],
    "f50": [dict(column="val1", r=1.5)],
    "f51": [dict(column="val1", )],
    "f52": [dict(column="val1", )],
    "f53": [dict(column="val1", )],
    "f54": [dict(column="val1", coeff=2)],
    "f55": [dict(column="val1", )],
    "f56": [dict(column="val1", )],
    "f57": [dict(column="val1", )],
    "f58": [dict(column="val1", )],
    "f59": [dict(column="val1", )],
    "f60": [dict(column="val1", r=.5)],
    "f61": [dict(column="val1", lag=3)],
    "f62": [dict(column="val1", value=1.9)],
    "f63": [dict(column="val1", )],
    "f64": [dict(column="val1", )],
}


def test_calc_all():
    res = calc_all(MULTIPLE_COLUMN_DF, PARAMS_SET, WINDOW)
    assert len(res) == len(MULTIPLE_COLUMN_DF) - WINDOW + 1


def test_calc_all_column_param():
    ps = {
        "f1": [{"column": "val2"}],
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
    "feature_key, feature_params",
    [
        (feature_key, {k: v for k, v in feature_params[0].items() if k != "column"})
        for feature_key, feature_params in PARAMS_SET.items()
    ],
    ids=list(PARAMS_SET.keys())
)
def test_features(feature_key, feature_params):
    res = FEATURE_FUNCS[feature_key](MULTIPLE_COLUMN_DF.val1.values, WINDOW, **feature_params)
    assert len(res) == len(MULTIPLE_COLUMN_DF) - WINDOW + 1
