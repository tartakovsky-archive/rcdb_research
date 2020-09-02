from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import rankdata
from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from . import proximity
from ..sampling import optimal_block_size
from ..sampling import bootstrap as run_bootstrap
from .checks import check_X_y_labels, check_clusters


class NMI(MetaEstimatorMixin, BaseEstimator):
    """
    Mutual Information

    Parameters
    ----------
    clusterer : Optional[AgglomerativeClustering]
        Instance of AgglomerativeClustering or None
    pooling_fn : Optional[Callable]
        Pooling function
    bootstrap : Optional[str]
        Bootstrap method, e.g 'iid', 'mbb', 'cbb', 'sbb'
    n_bootstraps : int
        Bootstrap repeats
    subsample_size : int
        Bootstrap subsample size
    random_state : int
        Seed for boostrap
    verbose : bool
        Show progress bar

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics._scorer import neg_log_loss_scorer
    >>> from sklearn.model_selection import KFold
    >>> from rcdb_research.feature_importance import cluster_ids_to_clusters, NMI
    >>> X = pd.DataFrame(dict(a=np.arange(100), b=-np.arange(100), c=[0.5, 0] * 50))
    >>> y = np.array([1, 0] * 50)
    >>> clusterer = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=0.75).fit(X.T)
    >>> clusters = cluster_ids_to_clusters(clusterer.labels_, X.columns)
    >>> imp = NMI()
    >>> _ = imp.fit(X, y, clusters=clusters)
    >>> imp.feature_importances_df_
    """
    def __init__(self,
                 clusterer: Optional[AgglomerativeClustering] = None,
                 pooling_fn: Optional[Callable] = None,
                 bootstrap: Optional[str] = 'sbb',
                 n_bootstraps: int = 10,
                 subsample_size: Optional[int] = None,
                 random_state=1,
                 verbose: bool = True):
        self.clusterer = clusterer
        self.pooling_fn = pooling_fn
        self.bootstrap = bootstrap
        self.n_bootstraps = n_bootstraps
        self.subsample_size = subsample_size
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.clusters = None
        self.feature_importances_ = None
        self.feature_importances_std_ = None
        self.feature_importances_rank_ = None
        self.feature_importances_labels_ = None
        self.feature_importances_df_ = None

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

        Returns
        -------
        Fitted estimator

        """
        X, y, labels, index = check_X_y_labels(X, y, labels)
        self.clusters = check_clusters(X, self.clusterer, clusters, labels)

        # If both clusters and poolin_fn are set, then feature agglomeration would be performed
        # Clusters would be merged into single features usign the pooling_fn
        shouldAgglomerate = self.clusters is not None and self.pooling_fn is not None
        if shouldAgglomerate:

            agg_X = pd.DataFrame(index=X.index)
            for i, cluster in enumerate(self.clusters):
                agg_X[cluster['name']] = self.pooling_fn(X[cluster['columns']])
                cluster['columns'] = [cluster['name']]
            X = agg_X

        if self.bootstrap is not None:
            block_size = optimal_block_size(data=y.values, method=self.bootstrap)
            subsamples = run_bootstrap(
                data=np.arange(y.size), method=self.bootstrap, block_size=block_size,
                subsample_size=self.subsample_size, repeats=self.n_bootstraps,
                seed=self.random_state.randint(2 ** 32 - 1), verbose=False
            )
        else:
            subsamples = [np.arange(y.size)]

        tqdm_ = tqdm if self.verbose else lambda x, *args, **kwargs: x

        results = {cluster['name']: [] for cluster in clusters}
        for sample_idx in tqdm_(subsamples, desc='NMI: processing bootstraps'):
            subs_X = X.iloc[sample_idx]
            subs_y = y.iloc[sample_idx]

            for cluster in clusters:
                results[cluster['name']].append(np.mean([
                    proximity.nmi(subs_X[col], subs_y) for col in cluster['columns']
                ]))

        importance = pd.DataFrame.from_dict(results)

        self.feature_importances_ = importance.mean().values
        self.feature_importances_std_ = importance.std().values
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
