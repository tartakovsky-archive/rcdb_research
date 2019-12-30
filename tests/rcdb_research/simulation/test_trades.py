import pytest
import numpy as np
import pandas as pd

from rcdb_research.simulation import Trades

from .test_converter import raises


TRADES_INDEX = pd.DatetimeIndex([
    '2014-06-04 01:17:00', '2014-06-04 01:27:00',
    '2014-06-04 01:34:00', '2014-06-04 01:52:00',
    '2014-06-04 02:04:00', '2014-06-04 02:17:00',
    '2014-06-04 02:59:00', '2014-06-04 03:23:00',
    '2014-06-04 13:58:00', '2014-06-04 14:27:00'
])


@pytest.fixture()
def trades():
    return Trades(
        directions=np.array([0, 0, 0, 0, 0, 1, 0, -1, -1, 0]),
        changes=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100, 0.0, -10, 10, 0.0]),
        fees=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, -0.1, -0.1, 0.0]),
        index=TRADES_INDEX.copy()
    )


@pytest.mark.parametrize(
    "trades_attr, params, test_result_values",
    (
        ('returns', {}, pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 99.0, 0.0, -10.1, 9.9, 0.0], index=TRADES_INDEX)),
        ('returns', dict(dense=True), pd.Series([99.0, -10.1, 9.9], index=TRADES_INDEX[[5, 7, 8]])),
        ('returns', dict(dense=True, raw=True), ([99.0, -10.1, 9.9], TRADES_INDEX[[5, 7, 8]])),
        ('profits', {}, pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 99.0, 0.0, 0.0, 9.9, 0.0], index=TRADES_INDEX)),
        ('profits', dict(dense=True), pd.Series([99.0, 9.9], index=TRADES_INDEX[[5, 8]])),
        ('profits', dict(dense=True, raw=True), ([99.0, 9.9], TRADES_INDEX[[5, 8]])),
        ('losses', {}, pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.1, 0.0, 0.0], index=TRADES_INDEX)),
        ('losses', dict(dense=True), pd.Series([-10.1], index=TRADES_INDEX[[7]])),
        ('losses', dict(dense=True, raw=True), ([-10.1], TRADES_INDEX[[7]])),
        ('equity', {}, pd.Series(
            [100.0, 100.0, 100.0, 100.0, 100.0, 5050.0, 5050.0, 4545.0, 5040.0, 5040.0], index=TRADES_INDEX)),
        ('equity', dict(dense=True), pd.Series([100, -405, -405.], index=TRADES_INDEX[[5, 7, 8]])),
        ('equity', dict(dense=True, raw=True), ([100, -405, -405.], TRADES_INDEX[[5, 7, 8]])),
        ('cum_return', {}, pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 49.5, 49.5, 44.45, 49.4, 49.4], index=TRADES_INDEX)),
        ('cum_return', dict(dense=True), pd.Series([0, -5.05, -5.05], index=TRADES_INDEX[[5, 7, 8]])),
        ('cum_return', dict(dense=True, raw=True), ([0, -5.05, -5.05], TRADES_INDEX[[5, 7, 8]])),
        ('drawdown', {}, pd.Series(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.09999999999999998, -0.001980198019801982, -0.001980198019801982],
            index=TRADES_INDEX
        )),
        ('drawdown', dict(dense=True), pd.Series([0, -5.05, -5.05], index=TRADES_INDEX[[5, 7, 8]])),
        ('drawdown', dict(dense=True, raw=True), ([0, -5.05, -5.05], TRADES_INDEX[[5, 7, 8]])),
        (
            'expected_cum_return',
            {},
            pd.Series([0.0, 4.94, 9.88, 14.82, 19.76, 24.7, 29.64, 34.58, 39.52, 44.46], index=TRADES_INDEX)
        ),
        (
            'expected_cum_return',
            dict(raw=True),
            ([0.0, 4.94, 9.88, 14.82, 19.76, 24.7, 29.64, 34.58, 39.52, 44.46], TRADES_INDEX)
        )

    )
)
def test_Trades_metrics(
    trades,
    trades_attr,
    params,
    test_result_values
):
    res = getattr(trades, trades_attr)(**params)

    if params.get('raw', False):
        assert np.array_equal(res[0], test_result_values[0])
        assert np.array_equal(res[1], test_result_values[1])
    else:
        assert test_result_values.equals(res)


@pytest.mark.parametrize(
    "metrics_name, params, test_result",
    (
        ('mean_profit', {}, 54.45),
        ('mean_profit', dict(after_fees=False), 55),
        ('mean_loss', {}, -10.1),
        ('mean_loss', dict(after_fees=False), -10),
        ('n_wins', {}, 2),
        ('n_wins', dict(after_fees=False), 2),
        ('n_losses', {}, 1),
        ('n_losses', dict(after_fees=False), 1),
        ('n_trades', {}, 3),
        ('pct_wins', {}, 2 / 3),
        ('pct_wins', dict(after_fees=False), 2 / 3),
        ('pct_losses', {}, 1 / 3),
        ('pct_losses', dict(after_fees=False), 1 / 3),
        ('tail_ratio', {}, 11.122222222222224),
        ('tail_ratio', dict(after_fees=False), 11.375000000000002),
        ('abs_return', {}, -5.05),
        ('abs_return', dict(after_fees=False), -5),
        ('max_drawdown', {}, -0.09999999999999998),
        ('max_drawdown', dict(after_fees=False), -0.0980392156862745),
        ('mar', {}, -37890.128388017125),
        ('mar', dict(after_fees=False), -38265.2781740371),
        ('n_bars', {}, 10),
        ('activity', {}, .3),
    )
)
def test_Trades_metrics_scalars(trades, metrics_name, params, test_result):
    assert getattr(trades, metrics_name)(**params) == test_result


@pytest.mark.parametrize(
    'input_name', ('directions', 'changes', 'fees',)
)
def test_Trades_init_validation(trades, input_name):
    params = dict(
        directions=trades.directions,
        changes=trades.changes,
        fees=trades.fees,
        index=trades.index
    )

    params[input_name] = params[input_name][:-1]

    raises(
        lambda: Trades(**params),
        f'{input_name}.size is not equal to index.size'
    )
