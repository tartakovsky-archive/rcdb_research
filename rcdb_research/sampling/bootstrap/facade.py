from typing import List, Optional, Dict
import logging
import numpy as np

from .wrappers import bootstrap, optimal_block_size

# Bootstrap ndarrays

bootstrap_1d = bootstrap

def bootstrap_2d(data: List[np.ndarray],
                 method: str,
                 block_size: Optional[int] = None,
                 subsample_size: int = None,
                 repeats: int = 100,
                 seed: int = None,
                 verbose: bool = True) -> List[List[np.ndarray]]:
    return [
        bootstrap(array, method, block_size, subsample_size, repeats, seed, verbose)
        for array in data
    ]

# Bootstrap {y_true, y_pred, index} dicts
def bootstrap_path(data: Dict[str, np.ndarray],
                   method: str,
                   block_size: Optional[int] = None,
                   subsample_size: int = None,
                   repeats: int = 100,
                   seed: int = None,
                   verbose: bool = True) -> List[Dict[str, np.ndarray]]:
    if block_size is None and method in ['mbb', 'cbb', 'sbb']:
        block_size = optimal_block_size(data['y_pred'], method)
        if verbose:
            logging.warning(
                f'\nParameter block_size is necessary for selected bootstrap method "{method}", but was not set'
                f'\nSetting block_size to optimal = {block_size:.2f}'
            )

    indices = np.arange(data['y_pred'].size)
    samples = bootstrap(indices, method=method, block_size=block_size,
                        subsample_size=subsample_size, repeats=repeats, seed=seed, verbose=verbose)
    resampled_paths = [
        {
            'y_true': data['y_true'][sample],
            'y_pred': data['y_pred'][sample],
            'index': data['index'][sample]
        }
        for sample in samples
    ]

    return resampled_paths

bootstrap_path_1d = bootstrap_path

def bootstrap_path_2d(data: List[Dict[str, np.ndarray]],
                      method: str,
                      block_size: Optional[int] = None,
                      subsample_size: int = None,
                      repeats: int = 100,
                      seed: int = None,
                      verbose: bool = True) -> List[List[Dict[str, np.ndarray]]]:
    return [
        bootstrap_path(path, method, block_size, subsample_size, repeats, seed, verbose)
        for path in data
    ]