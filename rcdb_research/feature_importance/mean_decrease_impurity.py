from typing import Callable, Optional

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from .utils import feature_importances
from ..sampling import optimal_block_size
from ..sampling import bootstrap as run_bootstrap
from .checks import check_X_y_labels, check_clusters


class MDI(MetaEstimatorMixin, BaseEstimator):
    def __init__(self,
                 estimator,
                 clusterer: Optional[AgglomerativeClustering] = None,
                 pooling_fn: Optional[Callable] = None,
                 bootstrap: Optional[str] = 'sbb',
                 n_bootstraps: int = 10,
                 subsample_size: Optional[int] = None,
                 random_state=1,
                 verbose: bool = True):
        self.estimator = estimator
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
        for sample_idx in tqdm_(subsamples, desc='MDI: processing bootstraps'):
            sw_train_dict = {'sample_weight': fit_sample_weight[sample_idx]} if fit_sample_weight is not None else {}

            self.estimator.fit(X=X.iloc[sample_idx], y=y.iloc[sample_idx], **sw_train_dict, **fit_params)

            impurity = feature_importances(self.estimator)

            for cluster in clusters:
                results[cluster['name']].append(np.mean([
                    impurity[np.where(X.columns == col)[0]] for col in cluster['columns']
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
