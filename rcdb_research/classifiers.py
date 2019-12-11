import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, threshold):
        """Replaces all features with `clf.predict_proba(X)`"""
        self.clf = clf
        self.threshold = threshold

    def fit(self, X, y, *args, **kwargs):
        self.clf.fit(X, y, *args, **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        predicts = self.clf.predict_proba(X, *args, **kwargs)[:, 1]
        predicts = np.where(predicts < self.threshold, 0, 1)
        return predicts

    def score(self, X, y, sample_weight=None):
        return self.clf.score(X, y, sample_weight)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['threshold'] = self.threshold
        params['clf'] = self.clf
        return params


class QuantileClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clf, q, quantile=0.5):
        self.clf = clf
        self.q = q
        self.quantile = quantile

    def fit(self, X, y):
        self.clf.fit(X, y)
        y_train_probas = self.clf.predict_proba(X)[:, 1]
        self.quantile = np.quantile(y_train_probas, self.q)
        return self

    def predict(self, X):
        y_test_probas = self.clf.predict_proba(X)[:, 1]
        predicts = np.where(y_test_probas > self.quantile, 1, 0)
        return predicts

    def score(self, X, y, sample_weight=None):
        return self.clf.score(X, y, sample_weight)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['clf'] = self.clf
        params['q'] = self.q
        params['quantile'] = self.quantile
        return params
