import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from rcdb_research.cross_validation import \
    WalkForwardCV, cross_val_predict_timeseries_splits, \
    split_indexes_to_bars, predict_splits, CombinatorialKFold, predicts_to_paths


@pytest.fixture
def walk_forward_cv():
    return WalkForwardCV(
        n_splits=10,
        test_size=0.3
    )


def test_walkforward_init(walk_forward_cv):
    assert walk_forward_cv.n_splits is not None
    assert walk_forward_cv.test_size is not None
    assert walk_forward_cv.gap_size is not None
    assert walk_forward_cv.expanding is not None


@pytest.mark.parametrize(
    "test_size, gap_size",
    [
        (0, 0.5),
        (0.5, 0.5),
        (0, 0),
        (1, 1),
    ]
)
def test_walkforward_init_wrong_sizes(test_size, gap_size):
    with pytest.raises(ValueError):
        WalkForwardCV(
            10,
            test_size=test_size,
            gap_size=gap_size
        )


def test_walkforward_get_n_split(walk_forward_cv):
    assert walk_forward_cv.n_splits == walk_forward_cv.get_n_splits()


@pytest.mark.parametrize(
    "expanding",
    [True, False]
)
def test_walkforward_split_expanding(
        expanding, n_splits=10, test_X=range(100)):
    cv = WalkForwardCV(
        n_splits=n_splits,
        test_size=0.3,
        expanding=expanding
    )

    splits = list(cv.split(test_X))
    assert len(splits) == n_splits
    assert splits[-1][-1][-1] == 99

    if expanding:
        assert all(split[0][0] == 0 for split in splits)
    else:
        assert splits[0][0][0] == 0
        assert all(split[0][0] != 0 for split in splits[1:])


def test_walkforward_split_not_enough_data(walk_forward_cv):
    with pytest.raises(ValueError):
        list(walk_forward_cv.split([]))


def test_walkforward_split_to_many_splits():
    with pytest.raises(ValueError):
        cv = WalkForwardCV(20, gap_size=0.3, test_size=0.5)
        list(
            cv.split(np.arange(20))
        )


@pytest.mark.parametrize(
    "params",
    [
        dict(
            estimator=DecisionTreeClassifier(),
            X=np.random.uniform(1, 10, (10, 2)),
            y=np.random.randint(0, 2, 10),
            n_jobs=1,
            cv=5
        ),
        dict(
            estimator=DecisionTreeClassifier(),
            X=np.random.uniform(1, 10, (10, 2)),
            y=np.random.randint(0, 2, 10),
            n_jobs=-1,
            cv=5
        )
    ]
)
def test_cross_val_predict_splits(params):
    assert len(cross_val_predict_timeseries_splits(**params))


@pytest.mark.parametrize(
    'cv_params, test_res',
    [
        (
            dict(n_splits=3, train_size=3, test_size=5, is_fixed=True),
            [
                ([2, 3, 4], [5, 6, 7, 8, 9]),
                ([7, 8, 9], [10, 11, 12, 13, 14]),
                ([12, 13, 14], [15, 16, 17, 18, 19])
            ]
        ),
        (
            dict(n_splits=3, train_size=5, test_size=5, is_fixed=True),
            [
                ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
                ([5, 6, 7, 8, 9], [10, 11, 12, 13, 14]),
                ([10, 11, 12, 13, 14], [15, 16, 17, 18, 19])
            ]

        ),
        (
            dict(n_splits=3, train_size=4, gap_size=1, test_size=5, is_fixed=True),
            [
                ([0, 1, 2, 3], [5, 6, 7, 8, 9]),
                ([5, 6, 7, 8], [10, 11, 12, 13, 14]),
                ([10, 11, 12, 13], [15, 16, 17, 18, 19])
            ]

        ),
    ]
)
def test_all_fixed_sizes_set(cv_params, test_res):
    cv = WalkForwardCV(**cv_params)
    res = list(cv.split(np.arange(20)))
    for i in range(cv_params['n_splits']):
        assert np.array_equal(test_res[i][0], res[i][0])
        assert np.array_equal(test_res[i][1], res[i][1])


def test_all_fixed_sizes_set_too_much_sizes():
    cv = WalkForwardCV(n_splits=3, test_size=6, train_size=5, is_fixed=True)

    with pytest.raises(ValueError) as ex:
        next(cv.split(np.arange(20)))
        assert ex.value == 'Provide more data'


@pytest.fixture(params=[False, True])
def is_numpy_input(request):
    return request.param


TEST_SPLIT_INPUT = dict(
    X=pd.DataFrame(np.array([2, 3, 4, 5, 6, 7, 8, 9])),
    y=pd.Series(-np.array([2, 3, 4, 5, 6, 7, 8, 9]))
)


@pytest.mark.parametrize(
    'class_params, test_res',
    [
        (
            dict(n_folds=2, embargo=None, tainted_up_to=None),
            [
                dict(
                    X_train=pd.DataFrame([6, 7, 8, 9], index=[4, 5, 6, 7]),
                    X_test=pd.DataFrame([2, 3, 4, 5], index=[0, 1, 2, 3]),
                    y_train=-pd.Series([6, 7, 8, 9], index=[4, 5, 6, 7]),
                    y_test=-pd.Series([2, 3, 4, 5], index=[0, 1, 2, 3]),
                ),
                dict(
                    X_train=pd.DataFrame([2, 3, 4, 5], index=[0, 1, 2, 3]),
                    X_test=pd.DataFrame([6, 7, 8, 9], index=[4, 5, 6, 7]),
                    y_train=-pd.Series([2, 3, 4, 5], index=[0, 1, 2, 3]),
                    y_test=-pd.Series([6, 7, 8, 9], index=[4, 5, 6, 7]),
                ),
            ]
        ),
        (
            dict(n_folds=2, embargo=2, tainted_up_to=None),
            [
                dict(
                    X_train=pd.DataFrame([8, 9], index=[6, 7]),
                    X_test=pd.DataFrame([2, 3, 4, 5], index=[0, 1, 2, 3]),
                    y_train=-pd.Series([8, 9], index=[6, 7]),
                    y_test=-pd.Series([2, 3, 4, 5], index=[0, 1, 2, 3]),
                ),
                dict(
                    X_train=pd.DataFrame([2, 3, 4, 5], index=[0, 1, 2, 3]),
                    X_test=pd.DataFrame([6, 7, 8, 9], index=[4, 5, 6, 7]),
                    y_train=-pd.Series([2, 3, 4, 5], index=[0, 1, 2, 3]),
                    y_test=-pd.Series([6, 7, 8, 9], index=[4, 5, 6, 7]),
                ),
            ]
        ),
        (
            dict(n_folds=2, embargo=None, tainted_up_to=1),
            [
                dict(
                    X_train=pd.DataFrame([2, 3, 7, 8, 9], index=[0, 1, 5, 6, 7]),
                    X_test=pd.DataFrame([4, 5, 6], index=[2, 3, 4]),
                    y_train=-pd.Series([2, 3, 7, 8, 9], index=[0, 1, 5, 6, 7]),
                    y_test=-pd.Series([4, 5, 6], index=[2, 3, 4]),
                ),
                dict(
                    X_train=pd.DataFrame([2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4]),
                    X_test=pd.DataFrame([7, 8, 9], index=[5, 6, 7]),
                    y_train=-pd.Series([2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4]),
                    y_test=-pd.Series([7, 8, 9], index=[5, 6, 7]),
                ),
            ]
        ),
        (
            dict(n_folds=2, embargo=2, tainted_up_to=1),
            [
                dict(
                    X_train=pd.DataFrame([2, 3, 9], index=[0, 1, 7]),
                    X_test=pd.DataFrame([4, 5, 6], index=[2, 3, 4]),
                    y_train=-pd.Series([2, 3, 9], index=[0, 1, 7]),
                    y_test=-pd.Series([4, 5, 6], index=[2, 3, 4]),
                ),
                dict(
                    X_train=pd.DataFrame([2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4]),
                    X_test=pd.DataFrame([7, 8, 9], index=[5, 6, 7]),
                    y_train=-pd.Series([2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4]),
                    y_test=-pd.Series([7, 8, 9], index=[5, 6, 7]),
                ),
            ]
        ),
        (
            dict(n_folds=2, embargo=None, tainted_up_to=5),
            [
                dict(
                    X_train=pd.DataFrame([2, 3, 4, 5, 6, 7, 9], index=[0, 1, 2, 3, 4, 5, 7]),
                    X_test=pd.DataFrame([8], index=[6]),
                    y_train=-pd.Series([2, 3, 4, 5, 6, 7, 9], index=[0, 1, 2, 3, 4, 5, 7]),
                    y_test=-pd.Series([8], index=[6]),
                ),
                dict(
                    X_train=pd.DataFrame([2, 3, 4, 5, 6, 7, 8], index=[0, 1, 2, 3, 4, 5, 6]),
                    X_test=pd.DataFrame([9], index=[7]),
                    y_train=-pd.Series([2, 3, 4, 5, 6, 7, 8], index=[0, 1, 2, 3, 4, 5, 6]),
                    y_test=-pd.Series([9], index=[7]),
                ),
            ]
        ),
        (
            dict(n_folds=1, embargo=2, tainted_up_to=4),
            [
                dict(
                    X_train=pd.DataFrame([2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4]),
                    X_test=pd.DataFrame([7, 8, 9], index=[5, 6, 7]),
                    y_train=-pd.Series([2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4]),
                    y_test=-pd.Series([7, 8, 9], index=[5, 6, 7]),
                )
            ]
        ),
        (
            dict(n_folds=1, embargo=None, tainted_up_to=4),
            [
                dict(
                    X_train=pd.DataFrame([2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4]),
                    X_test=pd.DataFrame([7, 8, 9], index=[5, 6, 7]),
                    y_train=-pd.Series([2, 3, 4, 5, 6], index=[0, 1, 2, 3, 4]),
                    y_test=-pd.Series([7, 8, 9], index=[5, 6, 7]),
                )
            ]
        ),
    ]
)
def test_CombinatorialKFold(class_params, test_res, is_numpy_input):
    input = TEST_SPLIT_INPUT

    if is_numpy_input:
        input = {k: v.T.values[0] if isinstance(v, pd.DataFrame) else v.values for k, v in input.items()}

    split_idxs = CombinatorialKFold(**class_params).split(**input)
    print(split_idxs)
    res = split_indexes_to_bars(indexes=split_idxs, **input)

    assert len(res) == len(test_res)
    for split, test_split in zip(res, test_res):
        for k, test_data in test_split.items():
            data = split[k]

            if test_data is None:
                assert data is None

            elif is_numpy_input:
                if isinstance(test_data, pd.Series):
                    assert np.array_equal(test_data.values, data)
                else:
                    assert np.array_equal(test_data.T.values[0], data)

            elif isinstance(test_data, pd.DataFrame) or isinstance(test_data, pd.Series):
                assert test_data.equals(data)


def print_splits(splits):
    for i, split in enumerate(splits):
        print('############################################')
        print('SPLIT #', i)
        print('###########')
        for k, v in split.items():
            print(k)
            print(v.to_string() if v is not None else None)
            print('_____')


@pytest.mark.parametrize(
    'class_params, input, error_msg',
    [
        (
            dict(n_folds=1, embargo=None, tainted_up_to=None),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'No data left for training set. '
            'Either set n_folds to > 1 or mark some data as tainted by setting tainted_up_to to not None'
        ),
        (
            dict(n_folds=0, embargo=None, tainted_up_to=None),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'No data left for training set. '
            'Either set n_folds to > 1 or mark some data as tainted by setting tainted_up_to to not None'
        ),
        (
            dict(n_folds=1, embargo=None, tainted_up_to=7),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'No data left for test set after separating tainted observations'
        ),
        (
            dict(n_folds=1, embargo=None, tainted_up_to=8),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'No data left for test set after separating tainted observations'
        ),
        (
            dict(n_folds=2, embargo=None, tainted_up_to=6),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'Not enough data left to perform 2 folds'
        ),
        (
            dict(n_folds=2, embargo=4, tainted_up_to=None),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'Not enough train set data to embargo'
        ),

    ]
)
def test_CombinatorialKFold_edge_cases(class_params, input, error_msg):
    with pytest.raises(CombinatorialKFold.SplitterException) as err:
        list(CombinatorialKFold(**class_params).split(**input))

    assert str(err.value) == error_msg


def test_CombinatorialKFold_edge_cases_with_cross_val_score():
    estimator = DecisionTreeRegressor()
    cv = CombinatorialKFold(n_folds=2, embargo=2, tainted_up_to=1)
    score = cross_val_score(estimator, TEST_SPLIT_INPUT['X'], TEST_SPLIT_INPUT['y'], cv=cv)
    assert len(score) == 2


def test_predict_splits(ohlcv_df):
    df = ohlcv_df.copy()
    df['y'] = np.random.randint(0, 2, len(df))
    y = df.y
    X = df.drop('y', 1)

    n = 5

    splits = split_indexes_to_bars(X, y, CombinatorialKFold(n, 10).split(X))

    res = predict_splits(DecisionTreeClassifier(), splits)

    assert len(res) == n

    for i in range(n):
        r = res[i]
        split = splits[i]
        assert np.array_equal(split['y_test'].values, r['y_true'])

        assert r['y_true'].shape == r['y_pred'].shape and r['y_true'].shape == r['index'].shape


@pytest.mark.parametrize(
    'params, x_size, ys_true',
    [
        (
            dict(n_folds=5, k_tests=3),
            15,
            [
                [0.01, 1.01, 2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01, 9.02, 10.02, 11.02, 12.03, 13.03, 14.03],
                [0.02, 1.02, 2.02, 3.02, 4.02, 5.02, 6.04, 7.04, 8.04, 9.04, 10.04, 11.04, 12.05, 13.05, 14.05],
                [0.03, 1.03, 2.03, 3.03, 4.03, 5.03, 6.05, 7.05, 8.05, 9.06, 10.06, 11.06, 12.06, 13.06, 14.06],
                [0.04, 1.04, 2.04, 3.07, 4.07, 5.07, 6.07, 7.07, 8.07, 9.07, 10.07, 11.07, 12.08, 13.08, 14.08],
                [0.05, 1.05, 2.05, 3.08, 4.08, 5.08, 6.08, 7.08, 8.08, 9.09, 10.09, 11.09, 12.09, 13.09, 14.09],
                [0.06, 1.06, 2.06, 3.09, 4.09, 5.09, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10, 12.10, 13.10, 14.10],
            ],
        ),
        (
            dict(n_folds=3, k_tests=2),
            15,
            [
                [0.01, 1.01, 2.01, 3.01, 4.01, 5.01, 6.01, 7.01, 8.01, 9.01, 10.02, 11.02, 12.02, 13.02, 14.02],
                [0.02, 1.02, 2.02, 3.02, 4.02, 5.03, 6.03, 7.03, 8.03, 9.03, 10.03, 11.03, 12.03, 13.03, 14.03],

            ]
        ),
        (
            dict(n_folds=3, k_tests=1),
            15,
            [
                [0.01, 1.01, 2.01, 3.01, 4.01, 5.02, 6.02, 7.02, 8.02, 9.02, 10.03, 11.03, 12.03, 13.03, 14.03],
            ]
        )
    ]
)
def test_predicts_to_paths(params, x_size, ys_true):
    preds = [
        dict(
            y_true=test_idxs + (i + 1) * .01,
            y_pred=test_idxs,
            index=test_idxs
        )
        for i, (_, test_idxs) in enumerate(CombinatorialKFold(**params).split(np.arange(x_size)))
    ]

    paths = predicts_to_paths(preds, **params)

    assert len(paths) == len(ys_true)
    for path, test_y_true in zip(paths, ys_true):
        test_y_true = np.array(test_y_true)
        test_path = {
            'y_true': test_y_true,
            'y_pred': test_y_true.astype(np.int),
            'index': test_y_true.astype(np.int)
        }
        assert path.keys() == test_path.keys()
        for k in test_path:
            assert np.array_equal(path[k], test_path[k])
