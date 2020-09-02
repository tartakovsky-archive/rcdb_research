from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectorMixin

from ..feature_importance.checks import check_X_y_labels, check_clusters
from .select_k_best import SelectKBest


class EFS(SelectorMixin, BaseEstimator):
    """
    Ensemble feature selection

    Parameters
    ----------
    estimators : List[sklearn.base.BaseEstimator]
        List of feature importance estimators for ensemble
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
    >>> efs = EFS(estimators=[MDI(RandomForestClassifier(random_state=1))], k_features=1)
    >>> efs.fit_transform(X, y)[:5]
    array([[0.5],
           [0. ],
           [0.5],
           [0. ],
           [0.5]])
    """
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
