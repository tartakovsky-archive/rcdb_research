from typing import List, Tuple
from joblib import Parallel, delayed
from sklearn.base import clone, BaseEstimator

import numpy as np
import pandas as pd

from .cross_validators import WalkForwardCV


class MultiInputSplitter(WalkForwardCV):
    """
    Multi input version of WalkForwardCV.
    Initialization:
    :param n_splits: count of splits
    :param test_size: test set size
    :param train_size: train set size
    :param gap_size: gap size
    :param is_fixed: if True then use sizes (train_size, test_size, gap_size, ) as count of items
    :param drop_earlier_history: if True then cut history by base dataset
    """

    def __init__(
        self,
        n_splits,
        test_size=None,
        train_size=None,
        gap_size=0,
        is_fixed=False,
        drop_earlier_history=False
    ):
        super().__init__(
            n_splits,
            test_size=test_size,
            train_size=train_size,
            gap_size=gap_size,
            is_fixed=is_fixed,
            expanding=True
        )
        self.drop_earlier_history = drop_earlier_history

    def split(
        self,
        main: Tuple[pd.DataFrame, pd.Series],
        additional: List[Tuple[pd.DataFrame, pd.Series]] = tuple(),
        verbose: bool = False
    ) -> List[dict]:
        """
        split method
        :param main: general dataset, tuple of X and y
        :param additional: additional datasets, list of tuple of X and y
        :param verbose: print splits
        """
        X_main, y_main = main
        index = X_main.index

        splits = []
        for train_idx_poses, test_idx_poses in super().split(index):
            train_start, train_end = index[train_idx_poses[0]], index[train_idx_poses[-1]]
            test_start, test_end = index[test_idx_poses[0]], index[test_idx_poses[-1]]
            if verbose:
                print(f'train {len(splits)} {train_start}:{train_end}')
                print(f'test {len(splits)} {test_start}:{test_end}')

            Xs_additional, ys_additional = [], []
            for set_idx, train_set in enumerate(additional):
                X, y = train_set
                if self.drop_earlier_history:
                    Xs_additional.append(X[(train_start <= X.index) & (X.index <= train_end)])
                    ys_additional.append(y[(train_start <= y.index) & (y.index <= train_end)])
                else:
                    Xs_additional.append(X[X.index <= train_end])
                    ys_additional.append(y[y.index <= train_end])

                if verbose:
                    print('=' * 10, 'Split', len(splits), '=' * 10)
                    print('Train', len(splits))
                    print(f'X_train{set_idx}')
                    print(Xs_additional[-1])
                    print(f'y_train{set_idx}')
                    print(ys_additional[-1])

            if self.drop_earlier_history:
                X_train_split = X_main[(train_start <= X_main.index) & (X_main.index <= train_end)]
                y_train_split = y_main[(train_start <= y_main.index) & (y_main.index <= train_end)]
            else:
                X_train_split = X_main[X_main.index <= train_end]
                y_train_split = y_main[y_main.index <= train_end]

            X_test_split = X_main[(test_start <= X_main.index) & (X_main.index <= test_end)]
            y_test_split = y_main[(test_start <= y_main.index) & (y_main.index <= test_end)]

            if verbose:
                print('Test main', len(splits))
                print('X_train')
                print(X_train_split)
                print('y_train')
                print(y_train_split)
                print()
                print('Test main', len(splits))
                print('X_test')
                print(X_test_split)
                print('y_test')
                print(y_test_split)
                print('=' * 10, 'End Split', len(splits), '=' * 10)

            splits.append(
                dict(
                    main=dict(
                        X_train=X_train_split,
                        y_train=y_train_split,
                        X_test=X_test_split,
                        y_test=y_test_split

                    ),
                    additional=[dict(X_train=x, y_train=y) for x, y in zip(Xs_additional, ys_additional)]
                )
            )

        return splits


def aggregate_splits(splits: List[dict], pre_agg_transforms: 'BaseEstimator') -> List[Tuple]:
    """
    Aggregate splits data by some estimator
    :param splits: splits from MultiInputSplitter
    :param pre_agg_transforms: estimator for agg transformation
    :returns: aggregated splits
    """
    aggregated_splits = []

    for split in splits:
        # Parse splits
        main_dataset, additional_datasets = split.values()
        X_train, y_train, X_test, y_test = [df for df in main_dataset.values()]
        y_test = y_test.values

        add_X_trains = [d['X_train'] for d in additional_datasets]
        add_y_trains = [d['y_train'].values for d in additional_datasets]

        # Pre-transform main dataset
        pre_agg_transforms.fit(X_train)
        X_train = pre_agg_transforms.transform(X_train)
        X_test = pre_agg_transforms.transform(X_test)

        agg_X_trains = [X_train]
        agg_y_trains = [y_train]

        if len(add_X_trains) > 0:
            # Pre-transform additional datasets
            for i, X_tr in enumerate(add_X_trains):
                if X_tr.size > 0:
                    add_X_trains[i] = pre_agg_transforms.fit_transform(X_tr)

            agg_X_trains.append(np.vstack(add_X_trains))
            agg_y_trains.append(np.hstack(add_y_trains))

        # Aggregate
        aggregated_splits.append((
            np.vstack(agg_X_trains),
            np.hstack(agg_y_trains),
            X_test,
            y_test,
        ))

    return aggregated_splits


def predict_aggregated_splits(clf: 'BaseEstimator', agg_splits: List[Tuple],
                              predict_proba: bool = False, predict_train: bool = False, flatten: bool = True,
                              fit_args: dict = {}, predict_args: dict = {},
                              n_jobs: int = -1) -> Tuple[np.array, np.array]:
    """
    Aggregated splits prediction
    :param cls: predictor
    :agg_splits: aggregated splits
    :predict_proba: if True returns probabilities of 1 instead of binary output
    :predict_train: if True returns predictions for train set in addition to y_true, y_pred
    :flatten: if False returns predictions separated by splits, if True merges them into one array
    :param n_jobs: count of jobs for  joblib.Parallel
    :returns: (y_true, y_pred, y_train_pred) if predict_train else (y_true, y_pred)
    """
    def predict_split(clf, split, predict_proba, predict_train, fit_args, predict_args):
        X_train, y_train, X_test, y_test = split

        clf.fit(X_train, y_train, **fit_args)

        if predict_proba:
            y_pred = clf.predict_proba(X_test, **predict_args)[:, 1]
            if predict_train:
                y_train_pred = clf.predict_proba(X_train, **predict_args)[:, 1]
        else:
            y_pred = clf.predict(X_test, **predict_args)
            if predict_train:
                y_train_pred = clf.predict(X_train, **predict_args)

        return (y_test, y_pred, y_train, y_train_pred) if predict_train else (y_test, y_pred)

    parallel = Parallel(n_jobs=n_jobs)
    prediction_blocks = parallel(
        delayed(predict_split)(clone(clf), split, predict_proba,
                               predict_train, fit_args, predict_args)
        for split in agg_splits
    )

    ys_tuple = tuple(np.array(t) for t in zip(*prediction_blocks))

    if flatten:
        ys_tuple = tuple(np.concatenate(arr) for arr in ys_tuple)

    return ys_tuple
