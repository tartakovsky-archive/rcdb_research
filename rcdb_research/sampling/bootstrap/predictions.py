from typing import List, Dict, Optional
import numpy as np
import logging

from .general import optimal_block_size, bootstrap


def bootstrap_path(data: Dict[str, np.ndarray],
                   method: str,
                   block_size: Optional[int] = None,
                   subsample_size: int = None,
                   repeats: int = 100,
                   verbose: bool = True) -> List[Dict[str, np.ndarray]]:
    if block_size is None and method in ['mbb', 'cbb', 'sbb']:
        block_size = optimal_block_size(data['y_pred'], method)
        if verbose:
            logging.warning(
                f'\nParameter block_size is necessary for selected bootstrap method = {method}, but was not set'
                f'Setting block_size to optimal = {block_size}'
            )

    indices = np.arange(data['y_pred'].size)
    samples = bootstrap(indices, method=method, block_size=block_size,
                        subsample_size=subsample_size, repeats=repeats, verbose=verbose)
    resampled_paths = [
        {
            'y_true': data['y_true'][sample],
            'y_pred': data['y_pred'][sample],
            'index': data['index'][sample]
        }
        for sample in samples
    ]

    return resampled_paths


def bootstrap_path_2d(data: List[Dict[str, np.ndarray]],
                      method: str,
                      block_size: Optional[int] = None,
                      subsample_size: int = None,
                      repeats: int = 100,
                      verbose: bool = True) -> List[List[Dict[str, np.ndarray]]]:
    return [
        bootstrap_path(path, method, block_size, subsample_size, repeats, verbose)
        for path in data
    ]
