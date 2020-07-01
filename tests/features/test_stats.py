import pytest
import numpy as np

from rcdb_research.features import stats

WINDOW = 5


@pytest.mark.parametrize(
    "feature",
    [
        stats.cmean,
        stats.fkurtosis,
        stats.pkurtosis,
        stats.skewness
    ]
)
def test_stats_features(feature, ohlcv_df):
    res = feature(ohlcv_df.open.values, WINDOW)
    assert len(res) == len(ohlcv_df)

    assert np.isnan(res[:WINDOW - 1]).all()
    assert np.isreal(res[WINDOW - 1:]).all()
