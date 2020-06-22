import pytest
import numpy as np
import pandas as pd

from rcdb_research.simulation_bt import BtRcdbStrategy, get_trading_simulation, get_trading_simulation_2nd_exchange
from rcdb_research.simulation_bt.wrappers import bt_data_feed_factory, DataFeedMissingFieldsException
from rcdb_research.simulation import KellySizing, Bitfinex, Costs


def data_to_df(data):
    df_data = pd.DataFrame(data)
    df_data = df_data.set_index('datetime')
    df_data.index = pd.to_datetime(df_data.index)
    return df_data


BARS_DATA = [
    {
        "datetime": "2020-01-01 00:00",
        "open": 100,
        "high": 100,
        "low": 100,
        "close": 110,
        "volume": 1000,
        "signal": 0.65,
        "exp_win": 2,
        "exp_loss": 2,
    },
    {
        "datetime": "2020-01-01 00:01",
        "open": 115,
        "high": 125,
        "low": 105,
        "close": 120,
        "volume": 2000,
        "signal": 0.7,
        "exp_win": 2,
        "exp_loss": 2,
    },
    {
        "datetime": "2020-01-01 00:02",
        "open": 115,
        "high": 140,
        "low": 115,
        "close": 115,
        "volume": 12244,
        "signal": 0.4,
        "exp_win": 2,
        "exp_loss": 2,
    },
    {
        "datetime": "2020-01-01 00:03",
        "open": 105,
        "high": 105,
        "low": 80,
        "close": 90,
        "volume": 1000,
        "signal": 0.5,
        "exp_win": 2,
        "exp_loss": 2,
    },
    {
        "datetime": "2020-01-01 00:04",
        "open": 100,
        "high": 105,
        "low": 95,
        "close": 95,
        "volume": 1000,
        "signal": 0.5,
        "exp_win": 2,
        "exp_loss": 2,
    },
    {
        "datetime": "2020-01-01 00:05",
        "open": 100,
        "high": 105,
        "low": 95,
        "close": 95,
        "volume": 1000,
        "signal": 0.7,
        "exp_win": 2,
        "exp_loss": 2,
    },
    {
        "datetime": "2020-01-01 00:06",
        "open": 102,
        "high": 105,
        "low": 95,
        "close": 115,
        "volume": 1000,
        "signal": 0.7,
        "exp_win": 2,
        "exp_loss": 2,
    },
    {
        "datetime": "2020-01-01 00:07",
        "open": 115,
        "high": 105,
        "low": 125,
        "close": 110,
        "volume": 1000,
        "signal": 0.5,
        "exp_win": 2,
        "exp_loss": 2,
    },
]


@pytest.mark.parametrize(
    'data',
    (
        BARS_DATA,
    )
)
def test_Backtrader_Simulation_run(data):  # noqa
    df_data = data_to_df(data)

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.155 / 100,
        maker_fee=-0.2 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    bitfinex_fee = 0.155 / 100

    expected_profit = 0.015 - bitfinex_fee * 2
    expected_loss = 0.015 - bitfinex_fee * 2

    trades, df_sim = get_trading_simulation(
        df_data=df_data,
        sizing=KellySizing(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        use_worst_pnl=False
    )

    assert df_sim.balance.values[-1] == 737566.272018916


@pytest.mark.parametrize(
    'data',
    (
        BARS_DATA,
    )
)
def test_Backtrader_Simulation_custom_sizing(data):  # noqa
    class StrangeSizer(KellySizing):
        def size(self, proba, exp_win, exp_loss):
            print(proba, exp_win, exp_loss)
            return super().size(proba)

    class BtCustom(BtRcdbStrategy):
        def get_size(self):
            return self.sizing.size(
                proba=self.data0.signal[0],
                exp_win=self.data0.exp_win[0],
                exp_loss=self.data0.exp_loss[0]
            )

    df_data = data_to_df(data)

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.155 / 100,
        maker_fee=-0.2 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    bitfinex_fee = 0.155 / 100

    expected_profit = 0.01 - bitfinex_fee * 2
    expected_loss = 0.01 - bitfinex_fee * 2

    trades, df_sim = get_trading_simulation(
        df_data=df_data,
        sizing=StrangeSizer(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        bt_strategy=BtCustom
    )

    assert df_sim.balance.values[-1] == 693663.1791642394


@pytest.mark.parametrize(
    'data',
    (
        BARS_DATA,
    )
)
def test_Backtrader_Simulation_risk_management(data):  # noqa
    df_data = data_to_df(data)

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.155 / 100,
        maker_fee=-0.2 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    bitfinex_fee = 0.155 / 100

    expected_profit = 0.015 - bitfinex_fee * 2
    expected_loss = 0.015 - bitfinex_fee * 2

    def max_dd_risk_manager(max_dd_allowed):
        def wrapper(self, desired_exposure):
            portfolio_value = self.broker.get_value()
            try:
                if portfolio_value > self.max_portfolio_value:
                    self.max_portfolio_value = portfolio_value

                elif 1 - portfolio_value / self.max_portfolio_value > max_dd_allowed:
                    return 0
            except AttributeError:
                self.max_portfolio_value = portfolio_value

            return desired_exposure
        return wrapper

    trades, df_sim = get_trading_simulation(
        df_data=df_data,
        sizing=KellySizing(win_size=expected_profit, loss_size=expected_loss, divider=10, direction='both'),
        exchange=exchange,
        bt_strategy=BtRcdbStrategy,
        risk_management_pre_trade=[max_dd_risk_manager(max_dd_allowed=0.3)]
    )

    print(df_sim)

    assert df_sim.balance.values[-1] == 737100.2237928074


def test_Data_Feed_Factory():  # noqa
    df_data = data_to_df(BARS_DATA)

    with pytest.raises(DataFeedMissingFieldsException):
        bt_data_feed_factory(df_data[['close', 'open']])

    data_feed_cls = bt_data_feed_factory(df_data)

    assert set(['exp_win', 'exp_loss']) - set(data_feed_cls.datafields) == set()


@pytest.mark.parametrize(
    'data',
    (
        BARS_DATA,
    )
)
def test_Backtrader_Simulation_run_2nd_exchange_sanity_check(data):  # noqa
    """
    Base equity (e.g. predict and trade same exchange) should be the same as
    predict and trade same exchange but with wnd exchange interface
    :param data:
    :return:
    """
    df_data = data_to_df(data)

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.155 / 100,
        maker_fee=-0.2 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    bitfinex_fee = 0.155 / 100

    expected_profit = 0.015 - bitfinex_fee * 2
    expected_loss = 0.015 - bitfinex_fee * 2

    trades, df_sim = get_trading_simulation_2nd_exchange(
        df_base=df_data,
        proba_arr=np.array([0.5] + list(df_data.signal.values)),
        df_trade=df_data,
        sizing=KellySizing(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        use_worst_pnl=False
    )

    assert df_sim.balance.values[-1] == 737566.272018916
