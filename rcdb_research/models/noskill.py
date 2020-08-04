from functools import partial

from sklearn.dummy import DummyClassifier
from sklearn.base import ClassifierMixin, BaseEstimator

SCORE_CLASSIFIER_MAPPING = dict(
    accuracy=partial(DummyClassifier, strategy='most_frequent'),
    f=partial(DummyClassifier, strategy='stratified'),
    roc_auc=partial(DummyClassifier, strategy='stratified'),
    avg_prec=partial(DummyClassifier, strategy='stratified'),
    log_loss=partial(DummyClassifier, strategy='prior'),
)


class NoSkillClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, strategy: str):
        if strategy not in SCORE_CLASSIFIER_MAPPING:
            raise Exception(f"Unsupported score. Choose one of {SCORE_CLASSIFIER_MAPPING.keys()}")
        self.strategy = strategy
        self.clf = SCORE_CLASSIFIER_MAPPING[strategy]()

    def fit(self, X, y, *args, **kwargs):
        return self.clf.fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        return self.clf.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        return self.clf.predict_proba(X, *args, **kwargs)

    def score(self, X, y, sample_weight=None):
        return self.clf.score(X, y, sample_weight=sample_weight)

    def get_params(self, deep=True):
        return {'strategy': self.strategy}
