from typing import List, Callable
import numpy as np


def score_2d(data: List[np.ndarray],
             fn: Callable[[np.ndarray], float]) -> List[float]:
    return [fn(array) for array in data]


def score_3d(data: List[List[np.ndarray]],
             fn: Callable[[np.ndarray], float]) -> List[List[float]]:
    return [
        [fn(array) for array in arrays]
        for arrays in data
    ]
