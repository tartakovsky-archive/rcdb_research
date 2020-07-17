from joblib import delayed, Parallel
from typing import List, Optional
import numpy as np

import logging
from ..sequential_bootstrap import sequential_bootstrap as seqb
from recombinator.optimal_block_length import optimal_block_length
from recombinator.block_bootstrap import moving_block_bootstrap, circular_block_bootstrap, stationary_bootstrap
from recombinator.iid_bootstrap import iid_bootstrap


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


def bootstrap(data: np.ndarray,
              method: str,
              block_size: Optional[int] = None,
              subsample_size: int = None,
              repeats: int = 100,
              seed: int = None,
              verbose: bool = True,
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

    rs = np.random.get_state()
    np.random.seed(seed)

    if method == 'mbb':
        samples = moving_block_bootstrap(data, block_size, repeats, subsample_size)
    elif method == 'cbb':
        samples = circular_block_bootstrap(data, block_size, repeats, subsample_size)
    elif method == 'sbb':
        samples = stationary_bootstrap(data, block_size, repeats, subsample_size)
    elif method == 'seqb':
        samples = sequential_bootstrap(data=data, t1=kwargs['t1'], bars_idx=kwargs['bars_idx'],
                                       repeats=repeats, subsample_size=subsample_size,
                                       n_jobs=kwargs.get('n_jobs', 1), verbose=verbose)
    else:  # iid
        samples = iid_bootstrap(data, repeats, subsample_size)

    np.random.set_state(rs)

    return list(samples)
