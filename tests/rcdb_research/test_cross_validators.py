import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from rcdb_research.cross_validators import WalkForwardCV, cross_val_predict_splits, CVResult


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
            n_jobs=1
        ),
        dict(
            estimator=DecisionTreeClassifier(),
            X=np.random.uniform(1, 10, (10, 2)),
            y=np.random.randint(0, 2, 10),
            n_jobs=-1
        )
    ]
)
def test_cross_val_predict_splits(params):
    assert len(cross_val_predict_splits(**params))


@pytest.fixture(scope='module')
def cv_result():
    return CVResult(
        y_pred=np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1]),
        y_true=np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 1]),
        index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )


def test_CVResult_init(cv_result):
    assert cv_result.y_true.size == 10
    assert cv_result.y_pred.size == 10
    assert cv_result.y_true_dense.size == 5
    assert cv_result.y_pred_dense.size == 5


def test_CVResult_head(cv_result: CVResult, n=8):
    cv_result_head = cv_result.head(n)
    assert (cv_result_head.y_true == cv_result.y_true[:n]).all()
    assert (cv_result_head.y_pred == cv_result.y_pred[:n]).all()
    assert (cv_result_head.index == cv_result.index[:n]).all()


def test_CVResult_tail(cv_result: CVResult, n=8):
    cv_result_tail = cv_result.tail(n)
    assert (cv_result_tail.y_true == cv_result.y_true[-n:]).all()
    assert (cv_result_tail.y_pred == cv_result.y_pred[-n:]).all()
    assert (cv_result_tail.index == cv_result.index[-n:]).all()


@pytest.mark.parametrize(
    "metric_name, params",
    [
        ("accuracy", dict(window=5)),
        ("accuracy", dict(window=5, sparse=False)),
        ("precision", dict(window=5)),
        ("precision", dict(window=5, sparse=False)),
        ("recall", dict(window=5)),
        ("recall", dict(window=5, sparse=False)),
        ("positives", dict()),
        ("positives", dict(sparse=False)),
        ("negatives", dict()),
        ("tp", dict()),
        ("fp", dict()),
        ("tn", dict()),
        ("fn", dict()),
    ]
)
def test_CVResult_metrics(cv_result, metric_name, params):
    v, idx = getattr(cv_result, metric_name)(**params)
    assert v.size != 0 and v.size == idx.size
    # print("\r\n Metric:", metric_name)
    # print(v)
    # print("\r\n IDs")
    # print(idx)

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
        assert ex.value is 'Provide more data'


def test_from_cv_results():
    cvp_results = [[
        np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1]),
        np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 1]),
    ]]
    index = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    cvres = CVResult.from_cross_val_predict_results(cvp_results, index)

    tp, idx = cvres.tp(sparse=False)

    assert list(tp) == [1, 0, 0, 1, 1]
    assert list(idx) == [2, 4, 7, 8, 10]
