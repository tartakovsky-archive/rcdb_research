import numpy as np

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sklearn.model_selection import cross_val_score


class RandomSelector(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            cv,
            estimator,
            n_features,
            n_iter,
            cv_scoring='accuracy',
            strategy=lambda scores: max(scores, key = lambda x: x["mean"])
    ):
        """
        Init method
        :param cv: cv instance for `cross_val_score`
        :param estimator: estimator for `cross_val_score`
        :param n_features: feature count`s for selector
        :param n_iter: count of selector iteration
        :param cv_scoring: `scoring` parameter for `cross_val_score`
        :param lambda strategy: function for select the best score. default is maximum value of means.
        """
        self.scores_ = []
        self.best_score_ = None

        self.cv = cv
        self.cv_scoring = cv_scoring
        self.estimator = estimator
        self.n_features = n_features
        self.n_iter = n_iter
        self.strategy = strategy

    @staticmethod
    def pd_to_np(data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values

        return data

    def _get_cross_validation_score(self, X, y, columns_indexes):
        scores = cross_val_score(
            self.estimator,
            cv=self.cv,
            X=X[:, columns_indexes],
            y=y,
            scoring=self.cv_scoring
        )

        return {
            'columns_indexes': columns_indexes,
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
        }

    def fit(self, X, y):
        X, y = self.pd_to_np(X), self.pd_to_np(y)
        self.scores_.clear()

        cols_idx = np.arange(X.shape[-1])
        prev_iters = set()

        for _ in range(self.n_iter):
            rnd_columns_indexes = np.random.choice(cols_idx, self.n_features, False)
            rnd_columns_indexes.sort()  # np.random.choice rearranged cols

            # disable duplicates
            str_rnd_columns = str(rnd_columns_indexes)
            if str_rnd_columns in prev_iters:
                continue

            prev_iters.add(str_rnd_columns)

            self.scores_.append(
                self._get_cross_validation_score(X, y, rnd_columns_indexes)
            )

        self.best_score_ = self.strategy(self.scores_)
        return self

    def transform(self, X):
        check_is_fitted(self, 'best_score_')
        return self.pd_to_np(X)[:, self.best_score_["columns_indexes"]]
