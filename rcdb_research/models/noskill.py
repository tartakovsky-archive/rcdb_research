import numpy as np

from functools import partial

from sklearn.dummy import DummyClassifier
from sklearn.base import ClassifierMixin, BaseEstimator


class LeastFrequentClassifier(DummyClassifier):
    def __init__(self):
        super().__init__(strategy='constant')

    def fit(self, X, y, sample_weight=None):
        self.constant = np.argmin(np.bincount(y))
        return super().fit(X, y, sample_weight)


SCORE_CLASSIFIER_MAPPING = dict(
    accuracy=partial(DummyClassifier, strategy='most_frequent'),
    precision=partial(LeastFrequentClassifier),
    f=partial(LeastFrequentClassifier),
    auc=partial(DummyClassifier, strategy='stratified'),
    pwa=partial(DummyClassifier, strategy='most_frequent'),
    neg_log_loss=partial(DummyClassifier, strategy='prior')
)


class NoSkillClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, score: str):
        if score not in SCORE_CLASSIFIER_MAPPING:
            raise Exception(f"Unsupported score. Choose one of {SCORE_CLASSIFIER_MAPPING.keys()}")
        self.clf = SCORE_CLASSIFIER_MAPPING[score]()

    def fit(self, X, y, *args, **kwargs):
        return self.clf.fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self.clf.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return self.clf.predict_proba(X, *args, **kwargs)

    def score(self, X, y, sample_weight=None):
        return self.clf.score(X, y, sample_weight=None)

    def get_params(self, deep=True):
        return self.clf.get_params(deep=True)
