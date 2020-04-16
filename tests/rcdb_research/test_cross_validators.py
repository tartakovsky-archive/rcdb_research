import pytest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from rcdb_research.cross_validation import \
    WalkForwardCV, cross_val_predict_timeseries_splits, \
    EmbargoedKFoldSplitterWithTainting, split_indexes_to_bars


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
            dict(k_fold=2, embargo=None, tainted_up_to=None),
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
            dict(k_fold=2, embargo=2, tainted_up_to=None),
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
            dict(k_fold=2, embargo=None, tainted_up_to=1),
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
            dict(k_fold=2, embargo=2, tainted_up_to=1),
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
            dict(k_fold=2, embargo=None, tainted_up_to=5),
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
            dict(k_fold=1, embargo=2, tainted_up_to=4),
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
            dict(k_fold=1, embargo=None, tainted_up_to=4),
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
def test_EmbargoedKFoldSplitterWithTainting(class_params, test_res, is_numpy_input):
    input = TEST_SPLIT_INPUT

    if is_numpy_input:
        input = {k: v.T.values[0] if isinstance(v, pd.DataFrame) else v.values for k, v in input.items()}

    split_idxs = EmbargoedKFoldSplitterWithTainting(**class_params).split(**input)
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
            dict(k_fold=1, embargo=None, tainted_up_to=None),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'No data left for training set. '
            'Either set k_fold to > 1 or mark some data as tainted by setting tainted_up_to to not None'
        ),
        (
            dict(k_fold=0, embargo=None, tainted_up_to=None),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'No data left for training set. '
            'Either set k_fold to > 1 or mark some data as tainted by setting tainted_up_to to not None'
        ),
        (
            dict(k_fold=1, embargo=None, tainted_up_to=7),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'No data left for test set after separating tainted observations'
        ),
        (
            dict(k_fold=1, embargo=None, tainted_up_to=8),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'No data left for test set after separating tainted observations'
        ),
        (
            dict(k_fold=2, embargo=None, tainted_up_to=6),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'Not enough data left to perform 2 folds'
        ),
        (
            dict(k_fold=2, embargo=4, tainted_up_to=None),
            dict(X=pd.DataFrame([2, 3, 4, 5, 6, 7, 8, 9])),
            'Not enough data left to perform 2 folds'
        ),

    ]
)
def test_EmbargoedKFoldSplitterWithTainting_edge_cases(class_params, input, error_msg):
    with pytest.raises(EmbargoedKFoldSplitterWithTainting.SplitterException) as err:
        list(EmbargoedKFoldSplitterWithTainting(**class_params).split(**input))
    assert err.match(error_msg)


def test_EmbargoedKFoldSplitterWithTainting_edge_cases_with_cross_val_score():
    estimator = DecisionTreeRegressor()
    cv = EmbargoedKFoldSplitterWithTainting(k_fold=2, embargo=2, tainted_up_to=1)
    score = cross_val_score(estimator, TEST_SPLIT_INPUT['X'], TEST_SPLIT_INPUT['y'], cv=cv)
    assert len(score) == 2
