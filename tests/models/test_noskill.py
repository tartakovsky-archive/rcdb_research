import pytest
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_classification

from rcdb_research.models import NoSkillClassifier
from rcdb_research.models.noskill import SCORE_CLASSIFIER_MAPPING


@pytest.mark.parametrize('score', SCORE_CLASSIFIER_MAPPING.keys())
def test_classifier(score, ):
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
    clf = NoSkillClassifier(score)
    clf.fit(X[:80], y[:80])

    assert isinstance(clf.clf, DummyClassifier)
    assert clf.predict(X[80:]).shape == (20,)
    assert clf.predict_proba(X[80:]).shape == (20, 2)


def test_classifier_wrong_metric():
    with pytest.raises(Exception) as ex:
        NoSkillClassifier('some metric')

    ex.match('Unsupported score.')
