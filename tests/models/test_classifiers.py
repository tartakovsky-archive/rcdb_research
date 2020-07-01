from rcdb_research.models import classifiers
import pytest


@pytest.mark.parametrize("n_seeds", [1, 5])
def test_LGBMClassifierEnsemble(n_seeds):  # noqa
    common_config = dict(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=12,
        num_leaves=80,
        bagging_fraction=0.1,
        feature_fraction=1,
        bagging_freq=1,
    )
    cfg_copy = common_config.copy()
    ensemble = classifiers.LGBMClassifierEnsemble(common_config, n_seeds, 1)
    if n_seeds > 1:
        assert len(ensemble.estimators[0][1].estimators) == n_seeds
    assert len(ensemble.estimators) == 3
    assert cfg_copy == common_config  # check that it was not modified
