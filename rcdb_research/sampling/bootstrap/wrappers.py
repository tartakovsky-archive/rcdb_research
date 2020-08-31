import logging
import multiprocessing

from functools import wraps
from typing import List, Optional

import numpy as np
from joblib import delayed, Parallel
from recombinator.optimal_block_length import optimal_block_length
from recombinator.block_bootstrap import moving_block_bootstrap, circular_block_bootstrap, stationary_bootstrap
from recombinator.iid_bootstrap import iid_bootstrap

from ..sequential_bootstrap import sequential_bootstrap as seqb


def sequential_bootstrap(data: np.ndarray,
                         t1: np.ndarray,
                         bars_idx: np.ndarray,
                         subsample_size: int = None,
                         repeats: int = 100,
                         seed: int = None,
                         n_jobs=1,
                         verbose: bool = True):
    def run_seq_b(t1, bars_idx, subsample_size, seed):  # noqa
        indices = seqb(t1, bars_idx, subsample_size, seed)
        return data[indices]

    if n_jobs == 1:
        parallel, fn = list, run_seq_b
    else:
        parallel, fn = Parallel(n_jobs=n_jobs), delayed(run_seq_b)

    return parallel(
        fn(t1, bars_idx, subsample_size, seed)
        for _ in range(repeats)
    )


def optimal_block_size(data: np.ndarray, method: str) -> Optional[float]:
    supported_methods = ['iid', 'mbb', 'cbb', 'sbb']
    if method not in supported_methods:
        raise ValueError(
            f'{method} method is not supported. Should be one of the following: {supported_methods}'
        )

    if method in ['mbb', 'cbb']:
        return int(optimal_block_length(data)[0].b_star_cb)
    elif method == 'sbb':
        return optimal_block_length(data)[0].b_star_sb
    else:
        return None


def with_seed(seed):
    def inner(f):
        @wraps(f)
        def inner2(*args, **kwargs):
            rs = np.random.get_state()
            np.random.seed(seed)
            res = f(*args, **kwargs)
            np.random.set_state(rs)
            return res
        return inner2
    return inner


def bootstrap(data: np.ndarray,
              method: str,
              block_size: Optional[int] = None,
              subsample_size: int = None,
              repeats: int = 100,
              seed: int = None,
              verbose: bool = True,
              n_jobs: int = 1,
              **kwargs) -> List[np.ndarray]:
    supported_methods = ['iid', 'mbb', 'cbb', 'sbb', 'seqb']
    if method not in supported_methods:
        raise ValueError(
            f'"{method}" method is not supported. Should be one of the following: {supported_methods}'
        )
    if (method == 'iid' or method == 'seqb') and block_size is not None and verbose:
        logging.warning(
            f'block_size parameter is ignored for "iid" and "seqb" bootstraps'
        )
    if subsample_size is None:
        subsample_size = data.size
        if verbose:
            logging.warning(
                f'Parameter subsample_size was not set. Setting subsample_size to data.size = {data.size}'
            )
    if block_size is None and method in ['mbb', 'cbb', 'sbb']:
        block_size = optimal_block_size(data, method)
        if verbose:
            logging.warning(
                f'\nParameter block_size is necessary for selected bootstrap method "{method}", but was not set'
                f'\nSetting block_size to optimal = {block_size:.2f}'
            )

    tasks = []
    n_jobs_ = int(n_jobs if n_jobs > 0 else multiprocessing.cpu_count())
    chunksize = int(np.ceil(repeats / n_jobs_))
    for seed_ in np.random.RandomState(seed).randint(2**31, size=n_jobs_):
        if method == 'mbb':
            tasks.append(delayed(with_seed(seed_)(moving_block_bootstrap))(data, block_size, chunksize, subsample_size))
        elif method == 'cbb':
            tasks.append(delayed(with_seed(seed_)(circular_block_bootstrap))(
                data, block_size, chunksize, subsample_size))
        elif method == 'sbb':
            tasks.append(delayed(with_seed(seed_)(stationary_bootstrap))(data, block_size, chunksize, subsample_size))
        elif method == 'seqb':
            tasks.append(delayed(with_seed(seed_)(sequential_bootstrap))(
                data=data, t1=kwargs['t1'], bars_idx=kwargs['bars_idx'],
                repeats=chunksize, subsample_size=subsample_size,
                n_jobs=1, verbose=verbose
            ))
        else:  # iid
            tasks.append(delayed(with_seed(seed_)(iid_bootstrap))(data, chunksize, subsample_size))
    results = Parallel(n_jobs)(tasks)
    samples = [x for y in results for x in y]

    return list(samples)
