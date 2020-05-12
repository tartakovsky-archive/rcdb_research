from typing import List, Dict, Optional
import numpy as np

from .general import optimal_block_size, bootstrap


def bootstrap_path(data: Dict[str, np.ndarray],
                   method: str,
                   block_size: Optional[int] = None,
                   subsample_size: int = None,
                   repeats: int = 100,
                   verbose: bool = True) -> List[Dict[str, np.ndarray]]:
    block_size = block_size or optimal_block_size(data['y_pred'], method)

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
