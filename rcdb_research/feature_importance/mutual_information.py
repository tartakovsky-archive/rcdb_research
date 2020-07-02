import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import logging
from typing import List, Callable, Optional

from sklearn.utils import check_random_state
from sklearn.cluster import AgglomerativeClustering

from .utils import cluster_labels_to_clusters
from ..sampling import optimal_block_size
from ..sampling import bootstrap as run_bootstrap
from ..metrics import proximity


def nmi(X: pd.DataFrame,
        y: pd.Series,
        clusters: Optional[List[dict]] = None,
        clusterer: Optional[AgglomerativeClustering] = None,
        pooling_fn: Optional[Callable] = None,
        bootstrap: Optional[str] = 'sbb',
        n_bootstraps: int = 10,
        subsample_size: Optional[int] = None,
        random_state=1,
        sort: bool = True,
        raw: bool = False,
        verbose: bool = True):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('X must be a pd.DataFrame')
    if not isinstance(y, pd.Series):
        raise ValueError('y must be a pd.Series')

    rs = check_random_state(random_state)

    # Flag to decide whether clusters should be agglomerated before scoring
    shouldAgglomerate = (clusters is not None or clusterer is not None) and pooling_fn is not None

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
    for sample_idx in tqdm_(subsamples, desc='NMI: processing bootstraps'):
        subs_X = X.iloc[sample_idx]
        subs_y = y.iloc[sample_idx]

        for cluster in clusters:
            results[cluster['name']].append(np.mean([
                proximity.nmi(subs_X[col], subs_y) for col in cluster['columns']
            ]))

    importance = pd.DataFrame.from_dict(results)
    if raw:
        return importance

    df = pd.concat({'mean': importance.mean(), 'std': importance.std()}, axis=1)
    df['rank'] = df['mean'].rank(method='first', ascending=False).astype(int)
    if sort:
        df = df.sort_values(by='mean', ascending=False)

    return df
