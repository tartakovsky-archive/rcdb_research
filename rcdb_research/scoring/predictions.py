from typing import List, Dict, Callable
import numpy as np


def score_path_1d(data: Dict[str, np.ndarray],
                  score: Callable[[np.ndarray, np.ndarray], float]) -> float:
    return score(data['y_true'], data['y_pred'])


def score_path_2d(data: List[Dict[str, np.ndarray]],
                  score: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    return np.array([score_path_1d(dct, score) for dct in data])


def score_path_3d(data: List[List[Dict[str, np.ndarray]]],
                  score: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    return np.array([score_path_2d(lst, score) for lst in data])
