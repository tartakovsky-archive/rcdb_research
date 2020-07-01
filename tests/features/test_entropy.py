import pytest
import numpy as np

from rcdb_research.features import entropy

TEST_WINDOW = 100


@pytest.mark.parametrize(
    "feature_func, feature_params",
    [
        (entropy.app_entropy, {}),
        (entropy.sample_entropy, {}),
        (entropy.spectral_entropy, {'sf': 10}),
        (entropy.svd_entropy, {}),
        (entropy.perm_entropy, {}),
        (entropy.binned_entropy, {'max_bins': 10})
    ]
)
def test_entropy(ohlcv_df, feature_func, feature_params):
    res = feature_func(ohlcv_df.open.values, window=TEST_WINDOW, **feature_params)
    assert len(res) == len(ohlcv_df)
    assert not np.isnan(res).all()
