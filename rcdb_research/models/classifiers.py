import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


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


def LGBMClassifierEnsemble(params, n_seeds=1, initial_seed=1): # noqa
    np.random.seed(initial_seed)
    common_config = params.copy()
    common_config.update({
        'random_seeds': np.random.randint(2**15, size=n_seeds),
        'type': 'lgbm'
    })
    ensemble = VotingClassifier(voting='soft', estimators=[
        ('gbdt', get_classifier({
            **common_config,
            'boosting': 'gbdt',
        })),
        ('rf', get_classifier({
            **common_config,
            'boosting': 'rf',
        })),
        ('goss', get_classifier({
            **common_config,
            'boosting': 'goss',
            'bagging_fraction': 1,
        })),
    ])
    return ensemble


def get_classifier(p):
    params = p.copy()

    clf_type = params.pop('type', None)
    threshold = params.pop('threshold', None)
    q = params.pop('q', None)
    random_seeds = params.pop('random_seeds', [1])

    supported_types = ['lgbm', 'xgb', 'rf', 'knn', 'mlp']
    if clf_type not in supported_types:
        raise ValueError(f"Please specify correct classifier type. Supported types = {supported_types}")

    if threshold is not None and q is not None:
        raise ValueError("Please set either threshold or q param. They are mutually exclusive")

    early_stopping_compatible = ['mlp']
    if 'early_stopping' in params and clf_type not in early_stopping_compatible:
        raise ValueError(f"Early stopping is not supported for {clf_type}")

    clfs = []
    if clf_type == 'lgbm':
        clfs = [LGBMClassifier(
            **params,
            random_state=seed,
            data_random_seed=seed,
            feature_fraction_seed=seed,
            objective_seed=seed,
            bagging_seed=seed,
            extra_seed=seed,
            drop_seed=seed
        ) for seed in random_seeds]
    elif clf_type == 'xgb':
        clfs = [XGBClassifier(**params, random_state=seed) for seed in random_seeds]
    elif clf_type == 'rf':
        clfs = [RandomForestClassifier(**params, random_state=seed) for seed in random_seeds]
    elif clf_type == 'knn':
        clfs = [KNeighborsClassifier(**params, random_state=seed) for seed in random_seeds]

    if len(random_seeds) > 1:
        clf = VotingClassifier([
            (f'{clf_type}({seed})', clf) for seed, clf in zip(random_seeds, clfs)
        ], 'soft')
    else:
        clf = clfs[0]

    if threshold is not None:
        clf = ThresholdClassifier(clf=clf, threshold=threshold)
    elif q is not None:
        clf = QuantileClassifier(clf=clf, q=q)

    return clf
