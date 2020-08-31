import numpy as np

from typing import Optional

from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectorMixin

from ..feature_importance.checks import check_X_y_labels, check_clusters

from .select_k_best import SelectKBest


class EFS(SelectorMixin, BaseEstimator):
    def __init__(self,
                 estimators: list,
                 k_features: int = 10,
                 clusterer: Optional[AgglomerativeClustering] = None):
        self.estimators = estimators
        self.k_features = k_features
        self.selectors = [SelectKBest(est, k_features=k_features) for est in estimators]
        self.clusterer = clusterer
        self.clusters = None
        self.labels = None

    def fit(self, X, y, clusters=None, labels=None, **fit_params):
        X, y, labels, index = check_X_y_labels(X, y, labels)
        self.labels = labels
        self.clusters = check_clusters(X, self.clusterer, clusters, self.labels)

        for sel in self.selectors:
            sel.fit(X, y, clusters, labels, **fit_params)

        return self

    def _get_support_mask(self):
        votes = [sel.get_support() for sel in self.selectors]

        mask = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0, arr=votes
        ).astype(bool)

        return mask
