from typing import List, Callable
import numpy as np


def score_1d(array: np.ndarray,
             fn: Callable[[np.ndarray], float]) -> float:
    return fn(array)


def score_2d(arrays: List[np.ndarray],
             fn: Callable[[np.ndarray], float]) -> np.ndarray:
    return np.array([fn(array) for array in arrays])


def score_3d(data: List[List[np.ndarray]],
             fn: Callable[[np.ndarray], float]) -> np.ndarray:
    return [score_2d(lst, fn) for lst in data]
