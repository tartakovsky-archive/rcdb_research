from typing import List, Optional
import numpy as np

import logging

from recombinator.optimal_block_length import optimal_block_length
from recombinator.block_bootstrap import moving_block_bootstrap, circular_block_bootstrap, stationary_bootstrap
from recombinator.iid_bootstrap import iid_bootstrap


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
              verbose: bool = True) -> List[np.ndarray]:
    supported_methods = ['iid', 'mbb', 'cbb', 'sbb']
    if method not in supported_methods:
        raise ValueError(
            f'"{method}" method is not supported. Should be one of the following: {supported_methods}'
        )
    if method == 'iid' and block_size is not None and verbose:
        logging.warning(
            f'block_size parameter is ignored for "iid" bootstrap'
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

    if method == 'mbb':
        samples = moving_block_bootstrap(data, block_size, repeats, subsample_size)
    elif method == 'cbb':
        samples = circular_block_bootstrap(data, block_size, repeats, subsample_size)
    elif method == 'sbb':
        samples = stationary_bootstrap(data, block_size, repeats, subsample_size)
    else:  # iid
        samples = iid_bootstrap(data, repeats, subsample_size)

    return list(samples)


def bootstrap_2d(data: List[np.ndarray],
                 method: str,
                 block_size: Optional[int] = None,
                 subsample_size: int = None,
                 repeats: int = 100,
                 verbose: bool = True) -> List[List[np.ndarray]]:
    return [
        bootstrap(array, method, block_size, subsample_size, repeats, verbose)
        for array in data
    ]
