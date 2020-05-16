import pytest
import pandas as pd

from rcdb_research.simulation_bt import BtRcdbStrategy, get_trading_simulation
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

    assert df_sim.balance.values[-1] == 737100.2237928074


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

    expected_profit = 0.015 - bitfinex_fee * 2
    expected_loss = 0.015 - bitfinex_fee * 2

    trades, df_sim = get_trading_simulation(
        df_data=df_data,
        sizing=StrangeSizer(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        bt_strategy=BtCustom
    )

    assert df_sim.balance.values[-1] == 737100.2237928074


def test_Data_Feed_Factory():  # noqa
    df_data = data_to_df(BARS_DATA)

    with pytest.raises(DataFeedMissingFieldsException):
        bt_data_feed_factory(df_data[['close', 'open']])

    data_feed_cls = bt_data_feed_factory(df_data)

    assert set(['exp_win', 'exp_loss']) - set(data_feed_cls.datafields) == set()
