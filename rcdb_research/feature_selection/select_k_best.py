import numpy as np

from typing import Optional

from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectorMixin

from ..feature_importance.checks import check_X_y_labels, check_clusters

from ..feature_importance.utils import feature_importances


class SelectKBest(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    def __init__(self,
                 estimator,
                 k_features=10,
                 clusterer: Optional[AgglomerativeClustering] = None):
        self.estimator = estimator
        self.k_features = k_features
        self.clusterer = clusterer
        self.clusters = None
        self.labels = None

    def fit(self, X, y, clusters=None, labels=None, **fit_params):
        X, y, labels, index = check_X_y_labels(X, y, labels)
        self.labels = labels
        self.clusters = check_clusters(X, self.clusterer, clusters, self.labels)

        self.estimator.fit(X, y, self.clusters, self.labels, **fit_params)
        return self

    def _get_support_mask(self):
        scores = feature_importances(self.estimator)
        selected_cluster_ids = np.argsort(-scores, kind='mergesort')[:self.k_features]

        cluster_mask = np.zeros_like(scores, dtype=bool)
        cluster_mask[selected_cluster_ids] = True

        selected_features = [ft
                             for cluster_idx in np.where(cluster_mask)[0]
                             for ft in self.clusters[cluster_idx]['columns']]
        feature_mask = np.isin(self.labels, selected_features)

        return feature_mask
