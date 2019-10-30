import pytest
import numpy as np
from sklearn.linear_model import SGDClassifier

from rcdb_research.classifiers import ThresholdClassifier

TEST_THRESHOLD = 0.5


@pytest.fixture
def threshold_clf():
    return ThresholdClassifier(
        clf=SGDClassifier(),
        threshold=TEST_THRESHOLD
    )


def test_ThresholdClassifier_init():
    inner_clf = SGDClassifier()
    clf = ThresholdClassifier(
        clf=inner_clf,
        threshold=TEST_THRESHOLD
    )

    assert clf.clf is inner_clf
    assert clf.threshold == TEST_THRESHOLD


def test_ThresholdClassifier_get_params(threshold_clf):
    params = threshold_clf.get_params()

    assert params['threshold'] == threshold_clf.threshold
    assert params['clf'] is threshold_clf.clf


def test_ThresholdClassifier_fit(threshold_clf):
    assert threshold_clf.fit(np.random.uniform(10, 20, (10, 2)), np.random.randint(0, 2, 10)) is threshold_clf


def test_ThresholdClassifier_score(threshold_clf):
    threshold_clf.fit(np.random.uniform(10, 20, (10, 2)), np.random.randint(0, 2, 10))
    assert threshold_clf.score(np.random.uniform(10, 20, (10, 2)), np.random.randint(0, 2, 10)) is not None
