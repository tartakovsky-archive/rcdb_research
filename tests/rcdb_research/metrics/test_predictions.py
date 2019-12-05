import pytest
import numpy as np
import pandas as pd

from rcdb_research.metrics.predictions import Predictions


@pytest.fixture(scope='module')
def predictions():
    return Predictions(
        y_pred=np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1]),
        y_true=np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 1]),
        index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )


def test_Predictions_init(predictions):
    assert predictions.y_true.shape[0] == 10
    assert predictions.y_pred.shape[0] == 10


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
def test_Predictions_init_wrong_inputs(params, exc_msg):
    with pytest.raises(ValueError) as ex:
        Predictions(**params)

    assert ex.match(exc_msg)


def test_Predictions_init_index_truncated(y=np.array([1, 2, 3]), index=np.array([0, 1, 2, 3, 4])):
    predictions = Predictions(
        y_true=y,
        y_pred=y,
        index=index
    )
    assert np.array_equal(index[-y.size:], predictions.index)


def test_Predictions_head(predictions: Predictions, n=8):
    result_head = predictions.head(n)

    assert np.array_equal(result_head.y_true, predictions.y_true[:n])
    assert np.array_equal(result_head.y_pred, predictions.y_pred[:n])


def test_Predictions_tail(predictions: Predictions, n=8):
    result_tail = predictions.tail(n)
    assert np.array_equal(result_tail.y_true, predictions.y_true[-n:])
    assert np.array_equal(result_tail.y_pred, predictions.y_pred[-n:])


def test_Predictions_tail_head_chain(predictions: Predictions, tail_n=3, head_n=10):
    cut_result = predictions.head(head_n).tail(tail_n)

    assert np.array_equal(predictions.y_pred[:head_n][-tail_n:], cut_result.y_pred)
    assert np.array_equal(predictions.y_true[:head_n][-tail_n:], cut_result.y_true)
    assert np.array_equal(predictions.index[:head_n][-tail_n:], cut_result.index)


@pytest.mark.parametrize(
    'metric_name, params, test_result',
    [
        ('activity', dict(), 0.5),
        ('activity', dict(window=5, raw=True), np.array([np.nan, np.nan, np.nan, np.nan, .4, .4, .4, .6, .4, .6])),
        (
            'activity',
            dict(window=5),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, np.nan, .4, .4, .4, .6, .4, .6]),
                np.arange(10)
            )
        ),
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
            dict(window=4, dense=False, raw=False),
            np.array([np.nan, np.nan, np.nan, .5, .5, 0, 0, 0.5, 0.5, 2 / 3])),
        (
            'precision',
            dict(window=4, dense=False, raw=True),
            np.array([np.nan, np.nan, np.nan, .5, .5, 0, 0, 0.5, 0.5, 2 / 3])
        ),
        (
            'precision',
            dict(window=4, dense=True, raw=True),
            pd.Series(
                np.array([np.nan, np.nan, np.nan, .5, .5]),
                np.array([1, 3, 6, 7, 9])
            )
        ),
        (
            'precision',
            dict(window=4, dense=True, raw=False),
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
def test_Predictions_metrics(predictions, metric_name, params, test_result):
    metric = getattr(predictions, metric_name)
    res = metric(**params)
    print(res)

    if not params:
        assert res == test_result
        return

    if 'window' in params:
        window = params['window']
        if params.get('raw', False):
            res, index = res
            assert np.array_equal(
                index,
                predictions.index if not params.get('dense', False) else test_result.index
            )

        if params.get('dense', False):
            assert len(test_result) == predictions.n_positives()

        assert np.isnan(res[:window - 1]).all()
        assert np.array_equal(res[window - 1:], test_result[window - 1:])


def test_Predictions_metrics_method(predictions):
    metrics = predictions.metrics()
    assert predictions.precision() == metrics.loc[0, 'precision']
    assert predictions.activity() == metrics.loc[0, 'activity']
    assert metrics.loc[0, 'observations'] == predictions.y_pred.size

    for column in metrics.columns:
        if column not in ['observations', 'activity', 'precision']:
            assert metrics.loc[0, column] == getattr(predictions, f'n_{column}')()
