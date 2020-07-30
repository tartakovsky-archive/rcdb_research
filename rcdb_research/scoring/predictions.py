from sklearn.calibration import calibration_curve
from typing import List, Dict, Callable, Tuple
import numpy as np
from joblib import Parallel, delayed


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


def calibrate_path(path, n_bins=20, strategy='quantile') -> Tuple[np.ndarray, np.ndarray]:
    true, pred = calibration_curve(
        path['y_true'], path['y_pred'], n_bins=n_bins, strategy=strategy
    )

    return true, pred


def calibrate_path_2d(list2d, n_bins=20, strategy='quantile', n_jobs=1) -> Tuple[np.ndarray, np.ndarray]:
    tasks = []
    for path in list2d:
        tasks.append(delayed(calibration_curve)(
            path['y_true'], path['y_pred'], n_bins=n_bins, strategy=strategy
        ))
    results = Parallel(n_jobs)(tasks)
    trues, preds = zip(*results)
    return np.hstack(trues), np.hstack(preds)


def calibrate_path_3d(list3d, n_bins=20, strategy='quantile', n_jobs=1) -> Tuple[np.ndarray, np.ndarray]:
    tasks = []
    for list2d in list3d:
        for path in list2d:
            tasks.append(delayed(calibration_curve)(
                path['y_true'], path['y_pred'], n_bins=n_bins, strategy=strategy
            ))
    results = Parallel(n_jobs)(tasks)
    trues, preds = zip(*results)
    return np.hstack(trues), np.hstack(preds)
