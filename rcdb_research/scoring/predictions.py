from typing import List, Dict, Callable
import numpy as np


def score_path(data: Dict[str, np.ndarray],
               fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
    return fn(data['y_true'], data['y_pred'])


def score_path_2d(data: List[Dict[str, np.ndarray]],
                  fn: Callable[[np.ndarray, np.ndarray], float]) -> List[float]:
    return [
        fn(d['y_true'], d['y_pred'])
        for d in data
    ]


def score_path_3d(data: List[List[Dict[str, np.ndarray]]],
                  fn: Callable[[np.ndarray, np.ndarray], float]) -> List[List[float]]:
    return [
        [fn(d['y_true'], d['y_pred']) for d in dicts]
        for dicts in data
    ]
