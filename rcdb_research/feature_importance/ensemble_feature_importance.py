import pandas as pd
import numpy as np
import logging

from typing import List, Set, Callable, Optional, Union

from sklearn.metrics import check_scoring
from scipy.stats import rankdata
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import BaseCrossValidator

from .mean_decrease_accuracy import mda
from .mutual_information import nmi
from .mean_decrease_impurity import mdi
from .utils import cluster_labels_to_clusters
from ..sampling.cv import CombinatorialCV

# Checks
from sklearn.utils import check_random_state
from .checks import check_X_y_labels, check_clusters
# ---

from sklearn.base import BaseEstimator


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


def efi(estimator,
        X: pd.DataFrame,
        y: pd.Series,
        clusters: Optional[List[dict]] = None,
        clusterer: Optional[AgglomerativeClustering] = None,
        pooling_fn: Optional[Callable] = None,
        methods: Set[str] = ('mda', 'mdi', 'nmi'),
        mda_cv: BaseCrossValidator = CombinatorialCV(5),
        mda_n_permutations: int = 10,
        mdi_bootstrap_method: str = 'sbb',
        mdi_n_boostraps: int = 10,
        mdi_reg_alphas: Optional[List[float]] = (0,),
        mdi_reg_lambdas: Optional[List[float]] = (0,),
        nmi_bootstrap_method: str = 'sbb',
        nmi_n_boostraps: int = 10,
        fit_params: dict = None,
        score_params: dict = None,
        scorer=None,
        random_state=1,
        sort: bool = True,
        full_report: bool = False,
        verbose: bool = True) -> Union[dict, pd.DataFrame]:
    if not isinstance(X, pd.DataFrame):
        raise ValueError('`X` must be a pd.DataFrame')
    if not isinstance(y, pd.Series):
        raise ValueError('`y` must be a pd.Series')
    if len(methods) == 0:
        raise ValueError('`methods` must contain at least one method')
    supported_methods = ['mda', 'mdi', 'nmi']
    unsupported_methods = list(set(methods) - set(supported_methods))
    if len(unsupported_methods) > 0:
        raise ValueError(
            f'{unsupported_methods} methods are not supported. Supported methods: {supported_methods}'
        )

    scorer = check_scoring(estimator, scorer)

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

    results = dict()
    if verbose:
        print('EFI: Scoring feature importances')
    for method in methods:
        if method == 'mda':
            mda_scores = mda(estimator=estimator, X=X, y=y, cv=mda_cv, clusters=clusters, clusterer=None,
                             pooling_fn=pooling_fn, fit_params=fit_params, score_params=score_params,
                             n_permutations=mda_n_permutations, scorer=scorer, random_state=random_state,
                             sort=False, verbose=verbose, raw=False)
            mda_scores['1/rank'] = 1 / mda_scores['rank']
            results['mda'] = mda_scores
        elif method == 'mdi':
            mdi_scores = mdi(estimator=estimator, X=X, y=y, clusters=clusters, clusterer=None,
                             pooling_fn=pooling_fn, bootstrap=mdi_bootstrap_method,
                             n_bootstraps=mdi_n_boostraps, fit_params=fit_params,
                             reg_alphas=mdi_reg_alphas, reg_lambdas=mdi_reg_lambdas,
                             random_state=random_state, sort=False, verbose=verbose, raw=False)
            mdi_scores['1/rank'] = 1 / mdi_scores['rank']
            results['mdi'] = mdi_scores
        elif method == 'nmi':
            nmi_scores = nmi(X=X, y=y, clusters=clusters, clusterer=None, pooling_fn=pooling_fn,
                             bootstrap=nmi_bootstrap_method, n_bootstraps=nmi_n_boostraps, random_state=random_state,
                             sort=False, verbose=verbose, raw=False)
            nmi_scores['1/rank'] = 1 / nmi_scores['rank']
            results['nmi'] = nmi_scores
        else:
            raise ValueError(f'{method} method is not supported. Supported methods: {supported_methods}')

    efi_scores = pd.DataFrame()

    efi_scores['mean'] = np.mean([res['1/rank'] for res in results.values()], axis=0)
    efi_scores['rank'] = efi_scores['mean'].rank(method='first', ascending=False).astype(int)
    efi_scores.index = [c['name'] for c in clusters]

    results['efi'] = efi_scores
    if sort:
        results = {k: v.sort_values(by='rank') for k, v in results.items()}

    return results if full_report else results['efi']
