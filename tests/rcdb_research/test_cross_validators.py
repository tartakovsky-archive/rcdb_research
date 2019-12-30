import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from rcdb_research.cross_validation.timeseries import WalkForwardCV, cross_val_predict_timeseries_splits


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
