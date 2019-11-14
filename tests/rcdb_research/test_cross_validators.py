import pytest
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from rcdb_research.cross_validators import WalkForwardCV, CVResult, cross_val_predict_timeseries_splits


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
    assert len(cross_val_predict_timeseries_splits(**params))


@pytest.fixture(scope='module')
def cv_result():
    return CVResult(
        y_pred=np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1]),
        y_true=np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 1]),
        index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )


def test_CVResult_init(cv_result):
    assert cv_result.y_true.shape[0] == 10
    assert cv_result.y_pred.shape[0] == 10


@pytest.mark.parametrize(
    'params, exc_msg',
    (
        (
            dict(y_true=np.array([1, 2]), y_pred=np.array([])),
            'Size of y_pred should be same size with y_true'
        ),
        (
            dict(y_true=np.array([1, 2]), y_pred=np.array([1, 2]), index=np.array([1])),
            'index.size=1 should be >= y_true.size=2'
        )
    ),
    ids=(
        'different sizes',
        'short index'
    )
)
def test_CVResult_init_wrong_inputs(params, exc_msg):
    with pytest.raises(ValueError) as ex:
        CVResult(**params)

    assert ex.match(exc_msg)


def test_CVResult_init_index_truncated(y=np.array([1, 2, 3]), index=np.array([0, 1, 2, 3, 4])):
    cv_result = CVResult(
        y_true=y,
        y_pred=y,
        index=index
    )
    assert np.array_equal(index[-y.size:], cv_result.y_true.index)
    assert np.array_equal(index[-y.size:], cv_result.y_pred.index)


def test_CVResult_head(cv_result: CVResult, n=8):
    cv_result_head = cv_result.head(n)

    assert np.array_equal(cv_result_head.y_true, cv_result.y_true[:n])
    assert np.array_equal(cv_result_head.y_pred, cv_result.y_pred[:n])


def test_CVResult_tail(cv_result: CVResult, n=8):
    cv_result_tail = cv_result.tail(n)
    assert np.array_equal(cv_result_tail.y_true, cv_result.y_true[-n:])
    assert np.array_equal(cv_result_tail.y_pred, cv_result.y_pred[-n:])


def test_CVResult_tail_head_chain(cv_result: CVResult, tail_n=3, head_n=10):
    cut_result = cv_result.head(head_n).tail(tail_n)

    assert np.array_equal(cv_result.y_pred[:head_n][-tail_n:], cut_result.y_pred)
    assert np.array_equal(cv_result.y_true[:head_n][-tail_n:], cut_result.y_true)

    assert np.array_equal(cv_result.y_true.index[:head_n][-tail_n:], cut_result.y_true.index)
    assert np.array_equal(cv_result.y_true.index[:head_n][-tail_n:], cut_result.y_true.index)


@pytest.mark.parametrize(
    'metric_name, params, test_result',
    [
        ('accuracy', dict(), 0.6),
        ('accuracy', dict(window=5, raw=True), np.array([np.nan, np.nan, np.nan, np.nan, .6, .6, .4, .4, .4, .6])),
        (
            'accuracy',
            dict(window=5),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, np.nan, .6, .6, .4, .4, .4, .6]),
                np.arange(10)
            )
        ),
        ('recall', dict(), 0.6),
        ('recall', dict(window=5, raw=True), np.array([np.nan, np.nan, np.nan, np.nan, .5, .5, .0, .5, 1 / 3, 2 / 3])),
        (
            'recall',
            dict(window=5, raw=False),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, np.nan, .5, .5, .0, .5, 1 / 3, 2 / 3]),
                np.arange(10)
            )
        ),
        ('precision', dict(), 0.6),
        (
            'precision',
            dict(window=4, sparse=True, raw=False),
            np.array([np.nan, np.nan, np.nan, .5, .5, 0, 0, 0.5, 0.5, 2 / 3])),
        (
            'precision',
            dict(window=4, sparse=True, raw=True),
            np.array([np.nan, np.nan, np.nan, .5, .5, 0, 0, 0.5, 0.5, 2 / 3])
        ),
        (
            'precision',
            dict(window=4, sparse=False, raw=True),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, .5, .5]),
                np.array([1, 3, 6, 7, 9])
            )
        ),
        (
            'precision',
            dict(window=4, sparse=False, raw=False),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, .5, .5]),
                np.array([1, 3, 6, 7, 9])
            )
        ),
        ('positives', dict(raw=True), np.array([0, 1, 0, -1, 0, 0, -1, 1, 0, 1])),
        (
            'positives',
            dict(raw=False),
            pd.Series(
                np.array([0, 1, 0, -1, 0, 0, -1, 1, 0, 1]),
                np.arange(10)
            )
        ),
        ('negatives', dict(raw=True), np.array([1, 0, 1, 0, -1, 1, 0, 0, -1, 0])),
        (
            'negatives',
            dict(raw=False),
            pd.Series(
                np.array([1, 0, 1, 0, -1, 1, 0, 0, -1, 0]),
                np.arange(10)
            )
        ),
        ('tp', dict(raw=True), np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1])),
        (
            'tp',
            dict(raw=False),
            pd.Series(
                np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1]),
                np.arange(10)
            )
        ),
        ('fp', dict(raw=True), np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])),
        (
            'fp',
            dict(raw=False),
            pd.Series(
                np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
                np.arange(10)
            )
        ),
        ('tn', dict(raw=True), np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
        (
            'tn',
            dict(raw=False),
            pd.Series(
                np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0]),
                np.arange(10)
            )
        ),
        ('fn', dict(raw=True), np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])),
        (
            'fn',
            dict(raw=False),
            pd.Series(
                np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0]),
                np.arange(10)
            )
        ),
        ('n_tp', dict(), 3),
        ('n_tp', dict(window=5, raw=True), np.array([np.nan, np.nan, np.nan, np.nan, 1, 1, 0, 1, 1, 2])),
        (
            'n_tp',
            dict(window=5, raw=False),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, np.nan, 1, 1, 0, 1, 1, 2]),
                np.arange(10)
            )
        ),
        ('n_fp', dict(), 2),
        ('n_fp', dict(window=5, raw=True), np.array([np.nan, np.nan, np.nan, np.nan, 1, 1, 2, 2, 1, 1])),
        (
            'n_fp',
            dict(window=5, raw=False),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, np.nan, 1, 1, 2, 2, 1, 1]),
                np.arange(10)
            )
        ),
        ('n_tn', dict(), 3),
        ('n_tn', dict(window=5, raw=True), np.array([np.nan, np.nan, np.nan, np.nan, 2, 2, 2, 1, 1, 1])),
        (
            'n_tn',
            dict(window=5, raw=False),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, np.nan, 2, 2, 2, 1, 1, 1]),
                np.arange(10)
            )
        ),
        ('n_fn', dict(), 2),
        ('n_fn', dict(window=5, raw=True), np.array([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2, 1])),
        (
            'n_fn',
            dict(window=5, raw=False),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2, 1]),
                np.arange(10)
            )
        ),
        ('n_negatives', dict(), 5),
        ('n_negatives', dict(window=3, raw=True), np.array([np.nan, np.nan, 2, 1, 2, 2, 2, 1, 1, 1])),
        (
            'n_negatives',
            dict(window=3, raw=False),
            pd.Series(
                np.array([np.nan, np.nan, 2, 1, 2, 2, 2, 1, 1, 1]),
                np.arange(10)
            )
        ),
        ('n_positives', dict(), 5),
        ('n_positives', dict(window=3, raw=True), np.array([np.nan, np.nan, 1, 2, 1, 1, 1, 2, 2, 2])),
        (
            'n_positives',
            dict(window=3, raw=False),
            pd.Series(
                np.array([np.nan, np.nan, 1, 2, 1, 1, 1, 2, 2, 2]),
                np.arange(10)
            )
        ),
    ]
)
def test_CVResult_metrics_(cv_result, metric_name, params, test_result):
    metric = getattr(cv_result, metric_name)
    res = metric(**params)

    if not params:
        assert res == test_result
        return

    if 'window' in params:
        window = params['window']
        if params.get('raw', False):
            res, index = res
            assert np.array_equal(
                index.values,
                cv_result.y_pred.index.values if params.get('sparse', True) else test_result.index.values
            )

        if not params.get('sparse', True):
            assert len(test_result) == cv_result.n_positives()

        assert np.isnan(res[:window - 1]).all()
        assert np.array_equal(res[window - 1:], test_result[window - 1:])


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
