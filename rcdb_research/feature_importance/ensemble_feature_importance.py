from typing import Optional

import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import check_random_state

from .checks import check_X_y_labels, check_clusters


class EFI(BaseEstimator):
    def __init__(self,
                 estimators: list,
                 clusterer: Optional[AgglomerativeClustering] = None,
                 random_state=1,
                 verbose=True):
        self.estimators = estimators
        self.clusterer = clusterer
        self.clusters = None
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.feature_importances_ = None
        self.feature_importances_std_ = None
        self.feature_importances_rank_ = None
        self.feature_importances_labels_ = None
        self.feature_importances_df_ = None

    def fit(self, X, y, clusters=None, labels=None, **fit_params):
        X, y, labels, index = check_X_y_labels(X, y, labels)
        self.clusters = check_clusters(X, self.clusterer, clusters, labels)

        for estimator in self.estimators:
            estimator.fit(X, y, self.clusters, labels, **fit_params)

        importances = pd.DataFrame([1 / e.feature_importances_rank_ for e in self.estimators])
        self.feature_importances_ = importances.mean().values
        self.feature_importances_std_ = importances.std().fillna(0).values
        self.feature_importances_rank_ = rankdata(-self.feature_importances_, method='dense').astype(int)
        self.feature_importances_labels_ = [c['name'] for c in self.clusters]
        self.feature_importances_df_ = pd.DataFrame.from_records(
            np.array([
                self.feature_importances_,
                self.feature_importances_std_,
                self.feature_importances_rank_
            ]).T,
            index=self.feature_importances_labels_,
            columns=['mean', 'std', 'rank']
        )
        return self
