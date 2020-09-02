import inspect
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import rankdata
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state
from sklearn.model_selection import check_cv
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from .checks import check_X_y_labels, check_clusters


class MDA(MetaEstimatorMixin, BaseEstimator):
    """
    Mean Decrease Accuracy

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Instance of sklearn estimator
    scorer : Callable
        Sklearn scorer
    cv : sklearn.base.BaseCrossValidator
        Instance of cross validator
    clusterer : Optional[AgglomerativeClustering]
        Instance of AgglomerativeClustering or None
    pooling_fn : Optional[Callable]
        Pooling function
    n_permutations : int
        Count of feature permutation in each cluster
    random_state : int
        Random state for shuffle function
    verbose : bool
        Show progress bar

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics._scorer import neg_log_loss_scorer
    >>> from sklearn.model_selection import KFold
    >>> from rcdb_research.feature_importance import cluster_ids_to_clusters, MDA, MDI, NMI
    >>> X = pd.DataFrame(dict(a=np.arange(100), b=-np.arange(100), c=[0.5, 0] * 50))
    >>> y = np.array([1, 0] * 50)
    >>> clusterer = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=0.75).fit(X.T)
    >>> clusters = cluster_ids_to_clusters(clusterer.labels_, X.columns)
    >>> m1_clf = RandomForestClassifier(random_state=1)
    >>> imp = MDA(m1_clf, neg_log_loss_scorer, KFold(n_splits=2), random_state=1)
    >>> _ = imp.fit(X, y, clusters=clusters)
    >>> imp.feature_importances_df_
             mean       std  rank
    a+0  0.000000  0.000000   2.0
    b+0  0.000000  0.000000   2.0
    c+0  0.842368  0.134668   1.0
    """
    def __init__(self,
                 estimator,
                 scorer,  # sklearn.make_scorer
                 cv=None,
                 clusterer: Optional[AgglomerativeClustering] = None,
                 pooling_fn: Optional[Callable] = None,
                 n_permutations: int = 10,
                 random_state=1,
                 verbose: bool = True):
        self.estimator = estimator
        self.scorer = check_scoring(estimator, scorer)
        self.cv = check_cv(cv)
        self.clusterer = clusterer
        self.clusters = None
        self.pooling_fn = pooling_fn
        self.n_permutations = n_permutations
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.feature_importances_ = None
        self.feature_importances_std_ = None
        self.feature_importances_rank_ = None
        self.feature_importances_labels_ = None
        self.feature_importances_df_ = None

    def fit(self, X, y, clusters=None, labels=None, score_params=None, **fit_params):
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
        score_params : Optional[dict]
            Dict of score parameters
        fit_params : dict
            fit parameters

        Returns
        -------
        Fitted estimator
        """
        score_params = score_params or {}

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

        fit_sample_weight = fit_params.pop('sample_weight', None)
        score_sample_weight = score_params.pop('sample_weight', None)

        baseline_scores = []  # [n_folds] of floats
        feature_scores = [[] for _ in self.clusters]  # [n_folds] of [(n_features * n_permutations)]

        # Split data. Show progress bar if verbose
        splits = self.cv.split(X=X)
        enumerate_splits = enumerate(tqdm(splits, desc='MDA: processing splits')) if self.verbose else enumerate(splits)

        for i, (train, test) in enumerate_splits:  # for split
            sw_train_dict = {'sample_weight': fit_sample_weight[train]} if fit_sample_weight is not None else {}
            sw_test_dict = {'sample_weight': score_sample_weight[test]} if score_sample_weight is not None else {}

            # Train the model on split's train set
            if 'clusters' in inspect.getfullargspec(self.estimator.fit).args:
                self.estimator.fit(X=X.iloc[train], y=y.iloc[train],
                                   clusters=self.clusters, **sw_train_dict, **fit_params)
            else:
                self.estimator.fit(X=X.iloc[train], y=y.iloc[train],
                                   **sw_train_dict, **fit_params)

            # Get baseline score for split's test set
            baseline_score = self.scorer(self.estimator, X.iloc[test], y.iloc[test],
                                         **sw_test_dict, **score_params)
            baseline_scores.append(baseline_score)

            # Get scores for permuted features
            for j, cluster in enumerate(self.clusters):
                X_test = X.iloc[test].copy()

                for _ in range(self.n_permutations):
                    # Permute all features in the cluster
                    for col in cluster['columns']:
                        self.random_state.shuffle(X_test[col].values)

                    ft_score = self.scorer(self.estimator, X_test, y.values[test],
                                           **sw_test_dict, **score_params)
                    feature_scores[j].append(baseline_scores[i] - ft_score)

        self.feature_importances_ = np.mean(feature_scores, axis=1)
        self.feature_importances_std_ = np.std(feature_scores, axis=1, ddof=1)
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
