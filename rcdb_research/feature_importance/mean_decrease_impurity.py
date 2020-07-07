import pandas as pd
import numpy as np
import logging

from typing import List, Callable, Optional

from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm

from .utils import cluster_labels_to_clusters, feature_importances

from ..sampling import optimal_block_size
from ..sampling import bootstrap as run_bootstrap

# Checks
from sklearn.utils import check_random_state, check_X_y
# ---

from sklearn.base import clone, BaseEstimator, MetaEstimatorMixin
import itertools


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
        if labels is None:
            labels = X.columns if hasattr(X, 'columns') else list(range(X.shape[1]))
        index = X.index if hasattr(X, 'index') else list(range(X.shape[0]))
        X, y = check_X_y(X, y)
        X = pd.DataFrame(X, index=index, columns=labels)
        y = pd.Series(y, index=index)  # refactor if profiler shows it's too slow

        # If either clusterer or clusters are set, then feature clustering would be performad
        if self.clusterer is not None:
            if clusters is not None:
                logging.warning(f'`clusterer` param is set, overriding `clusters` param')
            self.clusterer.fit(X.T)
            self.clusters = cluster_labels_to_clusters(self.clusterer.labels_, X.columns)
        elif clusters is not None:
            self.clusters = clusters
        else:
            self.clusters = [dict(name=col, columns=[col]) for col in X.columns]

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
                data=np.arange(y.size), method=self.bootstrap, block_size=block_size, subsample_size=self.subsample_size,
                repeats=self.n_bootstraps, seed=self.random_state.randint(2 ** 32 - 1), verbose=False
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


def mdi(estimator,
        X: pd.DataFrame,
        y: pd.Series,
        clusters: Optional[List[dict]] = None,
        clusterer: Optional[AgglomerativeClustering] = None,
        pooling_fn: Optional[Callable] = None,
        bootstrap: Optional[str] = 'sbb',
        n_bootstraps: int = 10,
        subsample_size: Optional[int] = None,
        reg_alphas: Optional[List[float]] = (0,),
        reg_lambdas: Optional[List[float]] = (0,),
        fit_params: dict = None,
        random_state=1,
        sort: bool = True,
        raw: bool = False,
        verbose: bool = True):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('X must be a DataFrame')
    if not isinstance(y, pd.Series):
        raise ValueError('y must be a Series')

    skip_reg_params = False
    if getattr(estimator, 'reg_alpha', None) is not None and getattr(estimator, 'reg_lambda', None) is not None:
        def assign_alpha_lambda(estimator, reg_alpha, reg_lambda):
            m = clone(estimator)
            m.reg_alpha = reg_alpha
            m.reg_lambda = reg_lambda
            return m
    elif getattr(estimator, 'base_estimator', None) is not None and \
            getattr(estimator.base_estimator, 'reg_alpha', None) is not None and \
            getattr(estimator.base_estimator, 'reg_lambda', None) is not None:
        def assign_alpha_lambda(estimator, reg_alpha, reg_lambda):
            m = clone(estimator)
            m.base_estimator.reg_alpha = reg_alpha
            m.base_estimator.reg_lambda = reg_lambda
            return m
    else:
        logging.warning('Neither estimator nor estimator.base_estimator have reg_alpha / reg_lambda properties. '
                        'Ingoring reg_alphas / reg_lambdas input params')
        skip_reg_params = True

        def assign_alpha_lambda(estimator, reg_alpha, reg_lambda):
            m = clone(estimator)
            return m

    rs = check_random_state(random_state)
    fit_params = fit_params or {}

    # Flag to decide whether clusters should be agglomerated before scoring
    shouldAgglomerate = (clusters is not None or clusterer is not None) and pooling_fn is not None

    # Handle *_sample_weight in params to support sklearn.Pipelines
    sw_train_name, sw_train = next(
        (kv for kv in fit_params.items() if 'sample_weight' in kv[0]),
        (None, None)
    )
    _ = fit_params.pop(sw_train_name, None)

    # If clusterer is set, ignore clusters param and generate new clusters using clusterer
    # If clusters is set then the whole cluster would be mutated instead of a single feature
    # If clusters is None then each feature is put into separate cluster
    if clusterer is not None:
        if clusters is not None:
            logging.warning(f'`clusterer` param is set, ignoring `clusters` param')
        clusterer.fit(X.T)
        clusters = cluster_labels_to_clusters(clusterer.labels_, X.columns)
    else:
        clusters = clusters or [
            dict(name=col, columns=[col])
            for col in X.columns
        ]

    # If both clustered_subset and poolin_fn is set then feature agglomeration would be performed
    # Clusters would be merged into single features usign the pooling_fn
    if shouldAgglomerate:
        agg_X = pd.DataFrame(index=X.index)
        for i, cluster in enumerate(clusters):
            agg_X[cluster['name']] = pooling_fn(X[cluster['columns']].values)
            cluster['columns'] = [cluster['name']]
        X = agg_X

    if bootstrap is not None:
        block_size = optimal_block_size(data=y.values, method=bootstrap)
        subsamples = run_bootstrap(
            data=np.arange(y.size), method=bootstrap, block_size=block_size, subsample_size=subsample_size,
            repeats=n_bootstraps, seed=rs.randint(2 ** 32 - 1), verbose=False
        )
    else:
        subsamples = [np.arange(y.size)]

    tqdm_ = tqdm if verbose else lambda x, *args, **kwargs: x

    results = {cluster['name']: [] for cluster in clusters}
    for sample_idx in tqdm_(subsamples, desc='MDI: processing bootstraps'):
        for alpha, lam in itertools.product(reg_alphas, reg_lambdas) if not skip_reg_params else [[None, None]]:
            sw_train_dict = {sw_train_name: sw_train[sample_idx]} if sw_train_name is not None else {}
            model = assign_alpha_lambda(estimator, alpha, lam)
            model = model.fit(X=X.iloc[sample_idx], y=y.iloc[sample_idx], **sw_train_dict, **fit_params)

            impurity = feature_importances(model)

            for cluster in clusters:
                results[cluster['name']].append(np.mean([
                    impurity[np.where(X.columns == col)[0]] for col in cluster['columns']
                ]))

    importance = pd.DataFrame.from_dict(results)
    if raw:
        return importance

    df = pd.concat({'mean': importance.mean(), 'std': importance.std()}, axis=1)
    df['rank'] = df['mean'].rank(method='first', ascending=False).astype(int)
    if sort:
        df = df.sort_values(by='mean', ascending=False)

    return df
