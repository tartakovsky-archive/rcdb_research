import pandas as pd
import numpy as np
import logging

from typing import List, Set, Callable, Optional

from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import BaseCrossValidator
from tqdm.auto import tqdm

from .mean_decrease_accuracy import mda
from .mutual_information import nmi
from .mean_decrease_impurity import mdi
from .utils import cluster_labels_to_clusters
from ..sampling.cv import CombinatorialCV


def efs(estimator,
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
        verbose: bool = True):
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
    rs = check_random_state(random_state)

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

    results = {}
    if verbose:
        print('Scoring feature importances:')
    for method in methods:
        if method == 'mda':
            if verbose:
                print('Running MDA:')
            mda_scores = mda(estimator=estimator, X=X, y=y, cv=mda_cv, clusters=clusters, clusterer=None,
                             pooling_fn=pooling_fn, fit_params=fit_params, score_params=score_params,
                             n_permutations=mda_n_permutations, scorer=scorer, random_state=random_state,
                             sort=sort, verbose=verbose, raw=False)
            results['mda'] = mda_scores
        elif method == 'mdi':
            mdi_scores = mdi(estimator=estimator, X=X, y=y, clusters=clusters, clusterer=None,
                             pooling_fn=pooling_fn, bootstrap=mdi_bootstrap_method,
                             n_bootstraps=mdi_n_boostraps, fit_params=fit_params,
                             reg_alphas=mdi_reg_alphas, reg_lambdas=mdi_reg_lambdas,
                             random_state=random_state, sort=sort, verbose=verbose, raw=False)
            results['mdi'] = mdi_scores
        elif method == 'nmi':
            if verbose:
                print('Running NMI:')
            nmi_scores = nmi(X=X, y=y, clusters=clusters, clusterer=None, pooling_fn=pooling_fn,
                             bootstrap=nmi_bootstrap_method, n_bootstraps=nmi_n_boostraps, random_state=random_state,
                             sort=sort, verbose=verbose, raw=False)
            results['nmi'] = nmi_scores
        else:
            raise ValueError(f'{method} method is not supported. Supported methods: {supported_methods}')

    results['clusters'] = clusters

    return results
