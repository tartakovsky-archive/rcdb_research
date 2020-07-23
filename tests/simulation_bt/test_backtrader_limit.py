import pandas as pd

from rcdb_research.simulation_bt import get_trading_simulation
from rcdb_research.simulation import KellySizing, Bitfinex, Costs


def data_to_df(data):
    df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'signal'])
    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    return df


def test_limit_offset_min_long():
    """
    balance should be the same, not entry occurred
    """

    bt_data = data_to_df([
        # not entered
        ["2020-01-01 00:00", 2000, 2050, 1950, 2050, 1000, 0.7],
        ["2020-01-01 00:01", 2050, 2100, 2050, 2200, 1000, 0.5],
        ["2020-01-01 00:02", 2150, 2250, 2100, 2200, 1000, 0.5],

        # entry at 2049
        ["2020-01-01 00:00", 2000, 2050, 1950, 2050, 1000, 0.7],
        ["2020-01-01 00:01", 2049, 2100, 2050, 2200, 1000, 0.5],
        ["2020-01-01 00:02", 2150, 2250, 2100, 2200, 1000, 0.5],
    ])

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.075 / 100,
        maker_fee=0.025 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    expected_profit = 0.015 + exchange.costs.taker_fee * 2
    expected_loss = 0.015 + exchange.costs.taker_fee * 2

    trades, df_sim = get_trading_simulation(
        df_data=bt_data,
        sizing=KellySizing(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        use_worst_pnl=False,
        entry_limit=True,
        limit_offset_pct=0.0001,
        limit_offset_min=1,
    )
    df_sim = df_sim.set_index('datetime')
    df_sim[
        ['signal', 'open', 'high', 'low', 'close', 'volume']
    ] = bt_data[['signal', 'open', 'high', 'low', 'close', 'volume']]

    # no entry occurred on the first 0.7 signal
    assert df_sim.balance.values[0] == df_sim.balance.values[2]

    # entry occurred on the second 0.7 signal
    assert df_sim.balance.values[3] < df_sim.balance.values[5]
    # entry price 2049
    assert df_sim.target_price.values[4] == 2049.0


def test_limit_offset_min_short():
    """
    balance should be the same, not entry occurred
    """

    bt_data = data_to_df([
        # not entered
        ["2020-01-01 00:00", 2000, 2050, 1950, 2050, 1000, 0.3],
        ["2020-01-01 00:01", 2050, 2050, 2050, 1900, 1000, 0.5],
        ["2020-01-01 00:02", 1900, 2250, 2100, 2200, 1000, 0.5],

        # entry at 2049
        ["2020-01-01 00:00", 2000, 2050, 1950, 2050, 1000, 0.3],
        ["2020-01-01 00:01", 2049, 2100, 2050, 2200, 1000, 0.5],
        ["2020-01-01 00:02", 2150, 2250, 2100, 2200, 1000, 0.5],
    ])

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.075 / 100,
        maker_fee=0.025 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    expected_profit = 0.015 + exchange.costs.taker_fee * 2
    expected_loss = 0.015 + exchange.costs.taker_fee * 2

    trades, df_sim = get_trading_simulation(
        df_data=bt_data,
        sizing=KellySizing(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        use_worst_pnl=False,
        entry_limit=True,
        limit_offset_pct=0.0001,
        limit_offset_min=1,
    )
    df_sim = df_sim.set_index('datetime')
    df_sim[
        ['signal', 'open', 'high', 'low', 'close', 'volume']
    ] = bt_data[['signal', 'open', 'high', 'low', 'close', 'volume']]

    # no entry occurred on the first 0.7 signal
    assert df_sim.balance.values[0] == df_sim.balance.values[2]

    # entry occurred on the second 0.7 signal
    assert df_sim.balance.values[3] > df_sim.balance.values[5]
    # short position executed via limit order
    assert df_sim.exposure_current[4] < 0
    # entry price 2051 short
    assert df_sim.target_price.values[4] == 2051.0


def test_limit_offset_pct_long():
    """
    balance should be the same, not entry occurred
    """

    bt_data = data_to_df([
        # not entered
        ["2020-01-01 00:00", 2000, 2050, 1950, 2050, 1000, 0.7],
        ["2020-01-01 00:01", 2050, 2100, 2050, 2200, 1000, 0.5],
        ["2020-01-01 00:02", 2150, 2250, 2100, 2200, 1000, 0.5],

        # entry at 2049
        ["2020-01-01 00:00", 2000, 2050, 1950, 2050, 1000, 0.7],
        ["2020-01-01 00:01", 2000, 2100, 2050, 2200, 1000, 0.5],
        ["2020-01-01 00:02", 2150, 2250, 2100, 2200, 1000, 0.5],
    ])

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.075 / 100,
        maker_fee=0.025 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    expected_profit = 0.015 + exchange.costs.taker_fee * 2
    expected_loss = 0.015 + exchange.costs.taker_fee * 2

    trades, df_sim = get_trading_simulation(
        df_data=bt_data,
        sizing=KellySizing(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        use_worst_pnl=False,
        entry_limit=True,
        limit_offset_pct=0.01,
        limit_offset_min=1,
    )
    df_sim = df_sim.set_index('datetime')
    df_sim[
        ['signal', 'open', 'high', 'low', 'close', 'volume']
    ] = bt_data[['signal', 'open', 'high', 'low', 'close', 'volume']]

    # no entry occurred on the first 0.7 signal
    assert df_sim.balance.values[0] == df_sim.balance.values[2]

    # entry occurred on the second 0.7 signal
    assert df_sim.balance.values[3] < df_sim.balance.values[5]
    # entry price 2049
    assert df_sim.target_price.values[4] == 2029.5


def test_limit_offset_pct_short():
    """
    balance should be the same, not entry occurred
    """

    bt_data = data_to_df([
        # not entered
        ["2020-01-01 00:00", 2000, 2050, 1950, 2050, 1000, 0.3],
        ["2020-01-01 00:01", 2050, 2050, 2050, 1900, 1000, 0.5],
        ["2020-01-01 00:02", 1900, 2250, 2100, 2200, 1000, 0.5],

        # entry at 2049
        ["2020-01-01 00:00", 2000, 2050, 1950, 2050, 1000, 0.3],
        ["2020-01-01 00:01", 2049, 2100, 2050, 2200, 1000, 0.5],
        ["2020-01-01 00:02", 2150, 2250, 2100, 2200, 1000, 0.5],
    ])

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.075 / 100,
        maker_fee=0.025 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    expected_profit = 0.015 + exchange.costs.taker_fee * 2
    expected_loss = 0.015 + exchange.costs.taker_fee * 2

    trades, df_sim = get_trading_simulation(
        df_data=bt_data,
        sizing=KellySizing(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        use_worst_pnl=False,
        entry_limit=True,
        limit_offset_pct=0.01,
        limit_offset_min=1,
    )
    df_sim = df_sim.set_index('datetime')
    df_sim[
        ['signal', 'open', 'high', 'low', 'close', 'volume']
    ] = bt_data[['signal', 'open', 'high', 'low', 'close', 'volume']]

    # no entry occurred on the first 0.7 signal
    assert df_sim.balance.values[0] == df_sim.balance.values[2]

    # entry occurred on the second 0.7 signal
    assert df_sim.balance.values[3] > df_sim.balance.values[5]
    # short position executed via limit order
    assert df_sim.exposure_current[4] < 0
    # entry price 2051 short
    assert df_sim.target_price.values[4] == 2070.5
