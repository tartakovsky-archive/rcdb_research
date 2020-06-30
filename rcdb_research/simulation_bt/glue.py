import numpy as np
import pandas as pd
import backtrader as bt

from .wrappers import CommInfoFractional, bt_data_feed_factory, BtRcdbStrategy
from ..simulation.trades import Trades


def get_trading_simulation(
        df_data: pd.DataFrame,
        sizing,
        exchange,
        initial_cash: int = 1000000,
        use_worst_pnl=False,
        bt_strategy=BtRcdbStrategy,
        risk_management_pre_trade=None,
        entry_limit=False) -> (Trades, pd.DataFrame):
    """
    Do all the magic to configure backtrader, run simulation and get results.

    :param df_data: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'signal'])
    :param sizing: simulation.trades.Sizing
    :param exchange: simulation.trades.Exchange
    :param initial_cash: default=1.000.000 // Initial cash, don't change this until you know what are you doing.
    :param use_worst_pnl: default=False, if True will use worst intra bar price to evaluate equity
                         (low for longs and high for shorts)
    :param bt_strategy: backtrader strategy class
    :param risk_management_pre_trade: pre trader callback (modifies desired exposure before execution)
    :param entry_limit: increase position with limit orders
    :return:
    """
    cerebro = bt.Cerebro()
    cerebro.broker.addcommissioninfo(CommInfoFractional())
    cerebro.broker.setcommission(
        commission=-exchange.costs.taker_fee - exchange.costs.impact - exchange.costs.drift,
        leverage=1 / exchange.initial_margin
    )
    cerebro.broker.setcash(initial_cash)
    data = bt_data_feed_factory(df_data)(dataname=df_data)
    cerebro.adddata(data)

    # Add the strategy to cerebro
    cerebro.addstrategy(
        bt_strategy,
        sizing=sizing,
        use_worst_pnl=use_worst_pnl,
        risk_management_pre_trade=risk_management_pre_trade,
        entry_limit=entry_limit
    )

    # Analyzer
    # cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')

    strategies = cerebro.run()
    strat = strategies[0]

    df = pd.DataFrame(strat.story)

    trades_bt = Trades(
        index=pd.DatetimeIndex(df.datetime),
        balance=df.balance.values,
        unrealized_pnl=df.unrealized_pnl.values,
        exposure=df.exposure_current.values,
        context=None
    )

    return trades_bt, df
