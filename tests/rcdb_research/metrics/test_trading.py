import pytest
import numpy as np

from rcdb_research.simulation.trades.trades import TradingMetrics
from rcdb_research.simulation import TradingSimulator, Bitfinex, KellySizing, Costs
from rcdb_research.metrics.trading import equity, cum_return, drawdown, abs_return, cagr, mar


@pytest.fixture()
def trading_metrics(ohlcv_df):
    n = 1000

    df = ohlcv_df
    df['bid'] = df['open'] - 0.5
    df['ask'] = df['open'] + 0.5
    bidask = df[['bid', 'ask']].copy().shift(-1)[:-1][:1000]

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.155 / 100,
        maker_fee=-0.2 / 100,
        drift=-0.0 / 100,
        impact=-0.05 / 100,
    ))

    expected_profit = 0.5 + exchange.costs.taker_fee * 2
    expected_loss = -0.5 + exchange.costs.taker_fee * 2
    sizing_algo = KellySizing(win_size=expected_profit, loss_size=expected_loss, divider=10, direction='pos')
    ts = TradingSimulator(
        exchange=exchange,
        sizing_algo=sizing_algo
    )

    trades = ts.trades(np.random.rand(n), bidask)

    trades.metrics.__dict__['__trades_must_live_'] = trades
    return trades.metrics


def _cagr(trading_metrics):
    _abs_return = abs_return(cum_return(trading_metrics.returns().values))
    index = trading_metrics.returns().index
    years_passed = (index[-1] - index[0]).total_seconds() / 60 / 60 / 24 / 365.25
    return cagr(_abs_return, years_passed)


def _mar(trading_metrics):
    max_dd = drawdown(cum_return(returns=trading_metrics.returns().values)).min()
    return mar(_cagr(trading_metrics), max_dd)


def array_equal(a1, a2, p=10):
    return np.array_equal(np.round(a1, p), np.round(a2, p))


def float_equal(v1, v2, p=10):
    return round(v1, p) == round(v2, p)


@pytest.mark.parametrize(
    'method_name, metric_func',
    [
        ('equity', equity),
        ('cum_return', cum_return),
        ('drawdown', lambda returns: drawdown(cum_return(returns))),
    ]
)
def test_metrics_vec(trading_metrics: TradingMetrics, method_name, metric_func):
    returns = trading_metrics.returns().values

    res = metric_func(returns)
    test_res = getattr(trading_metrics, method_name)()
    assert array_equal(res, test_res)


def test_abs_return(trading_metrics):
    assert float_equal(
        abs_return(cum_return(trading_metrics.returns().values)),
        trading_metrics.total_return()
    )


def test_cagr(trading_metrics):
    assert float_equal(_cagr(trading_metrics), trading_metrics.cagr())


def test_mar(trading_metrics):
    assert float_equal(_mar(trading_metrics), trading_metrics.mar())
