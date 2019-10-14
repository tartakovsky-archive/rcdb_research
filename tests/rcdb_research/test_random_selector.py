from typing import Callable

import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from rcdb_research.feature_selection import RandomSelector


TEST_ESTIMATOR = DecisionTreeClassifier()
TEST_X = np.random.uniform(0, 5, (10, 10))
TEST_y = np.random.randint(0, 2, 10)


def test_init():
    rs = RandomSelector(cv=2, estimator=TEST_ESTIMATOR, n_features=5, n_iter=3)
    assert len(rs.scores_) == 0
    assert rs.best_score_ is None
    assert rs.cv == 2
    assert rs.estimator is TEST_ESTIMATOR
    assert rs.n_features == 5
    assert rs.n_iter == 3
    assert rs.cv_scoring == "accuracy"
    assert isinstance(rs.strategy, Callable)


@pytest.mark.parametrize(
    "X, y, n_iter, n_features",
    [
        (TEST_X, TEST_y, 5, 3),
        (pd.DataFrame(TEST_X), pd.Series(TEST_y), 3, 7),
    ]
)
def test_basic_usage(X, y, n_iter, n_features):
    rs = RandomSelector(cv=3, n_iter=n_iter, n_features=n_features, estimator=TEST_ESTIMATOR)
    res = rs.fit_transform(X, y)

    assert set(rs.best_score_.keys()) == {'columns_indexes', 'scores', 'mean', 'std', 'min', 'max'}
    assert np.array_equal(
        res,
        TEST_X[:, rs.best_score_["columns_indexes"]]
    )
    assert 0 < len(rs.scores_) <= n_iter


def test_in_pipeline():
    pipeline = Pipeline(
        [
            ("selector", RandomSelector(cv=3, n_iter=3, n_features=3, estimator=TEST_ESTIMATOR)),
            ("predictor", TEST_ESTIMATOR)
        ]
    )

    pipeline.fit(TEST_X, TEST_y)

    assert pipeline["selector"].best_score_ is not None
    assert 0 < len(pipeline["selector"].scores_)

    assert pipeline.predict(TEST_X) is not None
