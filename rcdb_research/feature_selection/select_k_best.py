from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectorMixin

from ..feature_importance.checks import check_X_y_labels, check_clusters
from ..feature_importance.utils import feature_importances


class SelectKBest(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """
    Select k-best features

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Feature importance estimator
    k_features : int
        Count of features to select
    clusterer : Optional[AgglomerativeClustering]
        Instance of AgglomerativeClustering

    Examples
    --------
    >>> import pandas as pd
    >>> from rcdb_research.feature_importance import MDI
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = pd.DataFrame(dict(a=np.arange(100), b=-np.arange(100), c=[0.5, 0] * 50))
    >>> y = np.array([1, 0] * 50)
    >>> efs = SelectKBest(MDI(RandomForestClassifier(random_state=1)), k_features=1)
    >>> efs.fit_transform(X, y)[:5]
    array([[0.5],
           [0. ],
           [0.5],
           [0. ],
           [0.5]])
    """
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
        """
        Fit method

        Parameters
        ----------
        X : np.ndarray
            Input parameters with shape (n_samples, n_features)
        y : np.ndarray
            Input targets with shape (n_samples,)
        clusters : np.ndarray
            Array of clusters data
        labels : np.ndarray
            Array of labels
        fit_params : dict
            fit parameters

        Returns
        -------
        Fitted estimator
        """
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
