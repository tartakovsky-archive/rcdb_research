import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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

    def predict_proba(self, X, *args, **kwargs):
        return self.clf.predict_proba(X, *args, **kwargs)

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

    def fit(self, X, y, *args, **kwargs):
        self.clf.fit(X, y, *args, **kwargs)
        y_train_probas = self.clf.predict_proba(X, *args, **kwargs)[:, 1]
        self.quantile = np.quantile(y_train_probas, self.q)
        return self

    def predict(self, X, *args, **kwargs):
        y_test_probas = self.clf.predict_proba(X, *args, **kwargs)[:, 1]
        predicts = np.where(y_test_probas > self.quantile, 1, 0)
        return predicts

    def predict_proba(self, X, *args, **kwargs):
        return self.clf.predict_proba(X, *args, **kwargs)

    def score(self, X, y, sample_weight=None):
        return self.clf.score(X, y, sample_weight)

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['clf'] = self.clf
        params['q'] = self.q
        params['quantile'] = self.quantile
        return params


class KerasToRcdbPipelineWrapper:
    def __init__(self, params):
        self.params = params

        self.keras_model = params['keras_model']
        self.keras_kwargs = params.copy()
        self.keras_kwargs.pop("keras_model")
        self.estimator = None

    def fit(self, X, y):
        self.estimator = self.keras_model(
            **{
                **self.keras_kwargs,
                "steps_per_epoch": int(X.shape[0] / self.keras_kwargs['batch_size'])
            }
        )

        self.estimator.fit(X, y)

        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def get_params(self, *args, **kwargs):
        return {"params": self.params}


def get_classifier(p):
    params = p.copy()

    clf_type = params.pop('type', None)
    threshold = params.pop('threshold', None)
    q = params.pop('q', None)

    supported_types = ['lgbm', 'xgb', 'rf', 'knn']
    if clf_type not in supported_types:
        raise ValueError(f"Please specify correct classifier type. Supported types = {supported_types}")

    if threshold is not None and q is not None:
        raise ValueError("Please set either threshold or q param. They are mutually exclusive")

    clf = None
    if clf_type == 'lgbm':
        clf = LGBMClassifier(**params)
    elif clf_type == 'xgb':
        clf = XGBClassifier(**params)
    elif clf_type == 'rf':
        clf = RandomForestClassifier(**params)
    elif clf_type == 'knn':
        clf = KNeighborsClassifier(**params)

    if threshold is not None:
        clf = ThresholdClassifier(clf=clf, threshold=threshold)
    elif q is not None:
        clf = QuantileClassifier(clf=clf, q=q)

    return clf
