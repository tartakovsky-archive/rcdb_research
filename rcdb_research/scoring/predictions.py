import multiprocessing

from sklearn.calibration import calibration_curve
from typing import List, Dict, Callable, Tuple
import numpy as np
from joblib import Parallel, delayed


def score_path(data: Dict[str, np.ndarray],
               fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
    return fn(data['y_true'], data['y_pred'])


def score_path_2d(data: List[Dict[str, np.ndarray]],
                  fn: Callable[[np.ndarray, np.ndarray], float], n_jobs: int = 1) -> List[float]:
    tasks = []
    for d in data:
        tasks.append(delayed(fn)(d['y_true'], d['y_pred']))
    return Parallel(n_jobs)(tasks)


def score_path_3d(data: List[List[Dict[str, np.ndarray]]],
                  fn: Callable[[np.ndarray, np.ndarray], float], n_jobs: int = 1) -> List[List[float]]:
    tasks = []
    for dicts in data:
        for d in dicts:
            tasks.append(delayed(fn)(d['y_true'], d['y_pred']))
    n_jobs_ = int(n_jobs if n_jobs > 0 else multiprocessing.cpu_count())
    result = Parallel(n_jobs)(tasks)

    # restore the nested structure:
    i = 0
    ret = []
    for dicts in data:
        tmp = []
        for d in dicts:
            tmp.append(result[i])
            i += 1
        ret.append(tmp)
    return ret


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
