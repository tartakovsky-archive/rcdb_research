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
    "abs_energy": [{}],
    "abs_sum_of_changes": [{"column": "val2"}],
    "agg_linear_trend": [dict(column="val1", chunk_len=3, attr="rvalue", f_agg="mean")],
    "ar_coefs": [dict(column="val1", coeff=0.5, k=2)],
    "adf": [dict(column="val1", attr="teststate")],
    "autocorr": [dict(column="val1", lag=3)],
    "c3": [dict(column="val1", lag=3)],
    "change_quantiles": [dict(column="val1", ql=.05, qh=.2, isabs=True, f_agg="mean")],
    "cid_ce": [dict(column="val1", normalize=True)],
    "count_above_mean": [dict(column="val1", )],
    "count_below_mean": [dict(column="val1", )],
    "energy_ratio_by_chunks": [dict(column="val1", num_segments=2, segment_focus=1)],
    "fft_agg_centroid": [dict(column="val1", )],
    "fft_agg_variance": [dict(column="val1", )],
    "fft_agg_skew": [dict(column="val1", )],
    "fft_agg_kurtosis": [dict(column="val1", )],
    "fft_coefficient": [dict(column="val1", coeff=0, attr="abs")],
    "index_mass_quantile": [dict(column="val1", q=.3)],
    "large_standard_deviation": [dict(column="val1", r=.25)],
    "linear_trend": [dict(column="val1", attr="pvalue")],
    "longest_strike_above_mean": [dict(column="val1", )],
    "longest_strike_below_mean": [dict(column="val1", )],
    "mean_second_derivative_central": [dict(column="val1", )],
    "number_cwt_peaks": [dict(column="val1", n=3)],
    "partial_autocorrelation": [dict(column="val1", lag=3)],
    "pct_reoccuring_uniq": [dict(column="val1", )],
    "pct_reoccuring_all": [dict(column="val1", )],
    "ratio_beyond_r_sigma": [dict(column="val1", r=1.5)],
    "uniq_to_all_ratio": [dict(column="val1", )],
    "spkt_welch_density": [dict(column="val1", coeff=2)],
    "sum_of_reoccuring_datapoints": [dict(column="val1", )],
    "sum_of_reoccuring_values": [dict(column="val1", )],
    "sum_values": [dict(column="val1", )],
    "symmetry_looking": [dict(column="val1", r=.5)],
    "time_reversal_asymmetry_statistic": [dict(column="val1", lag=3)],
    "var_lt_stdev": [dict(column="val1", )],
}


def test_calc_all():
    res = calc_all(MULTIPLE_COLUMN_DF, PARAMS_SET, WINDOW)
    assert len(res) == len(MULTIPLE_COLUMN_DF)


def test_calc_all_column_param():
    ps = {
        "abs_energy": [{"column": "val2"}],
    }
    res = calc_all(MULTIPLE_COLUMN_DF, ps, 3)
    assert {f"{PREFIX}_abs_energy_3_val2"} == set(res.columns)


@pytest.mark.parametrize(
    "ps",
    [
        {"abs_energy": [{"column": "val2"}, {"column": "val1"}]},
        {"abs_energy": [{"column": "__all__"}]},
        {"abs_energy": [{}]},
    ]
)
def test_calc_all_column_param__all__(ps):
    res = calc_all(MULTIPLE_COLUMN_DF, ps, 3)
    assert {f"{PREFIX}_abs_energy_3_val1", f"{PREFIX}_abs_energy_3_val2"} == set(res.columns)


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
    assert len(res) == len(MULTIPLE_COLUMN_DF)
