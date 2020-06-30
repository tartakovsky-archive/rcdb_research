import rcdb_research.sampling.bootstrap
import rcdb_research.metrics.proximity

from sklearn.utils import check_random_state
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def nmi(
        X, y,
        clusters=None, bootstrap='sbb', n_bootstraps=10, subsample_size=None, random_state=1, raw=False, verbose=True,
):
    if not isinstance(X, pd.DataFrame):
        raise ValueError('X must be a DataFrame')
    if not isinstance(y, pd.Series):
        raise ValueError('y must be a Series')

    rs = check_random_state(random_state)

    clusters = clusters or [
        dict(name=col, columns=[col])
        for col in X.columns
    ]

    if bootstrap is not None:
        block_size = rcdb_research.sampling.optimal_block_size(data=y.values, method=bootstrap)
        subsamples = rcdb_research.sampling.bootstrap(
            data=np.arange(y.size), method=bootstrap, block_size=block_size, subsample_size=subsample_size,
            repeats=n_bootstraps, seed=rs.randint(2 ** 32 - 1), verbose=False
        )
    else:
        subsamples = [np.arange(y.size)]

    if verbose:
        tqdm_ = tqdm
    else:
        tqdm_ = lambda x, *args, **kwargs: x

    results = {cluster['name']: [] for cluster in clusters}
    for sample_idx in tqdm_(subsamples, desc='bootstraps processed:'):
        subs_X = X.iloc[sample_idx]
        subs_y = y.iloc[sample_idx]

        for cluster in clusters:
            results[cluster['name']].append(np.mean([
                rcdb_research.metrics.proximity.nmi(subs_X[col], subs_y) for col in cluster['columns']
            ]))
    if raw:
        # TODO: make this consistent with MDA
        return results

    return pd.DataFrame({
        'mean': np.array(list(results.values())).mean(axis=1),
        'std': np.array(list(results.values())).std(axis=1)
    }, index=list(results.keys()))
