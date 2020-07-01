import pytest
import numpy as np
import pandas as pd

from rcdb_research.simulation_bt import BtRcdbStrategy, get_trading_simulation
from rcdb_research.simulation_bt.wrappers import bt_data_feed_factory, DataFeedMissingFieldsException
from rcdb_research.simulation import KellySizing, Bitfinex, Costs
from rcdb_research.datasets.config import consolidate_datasets, add_basic_features


def data_to_df(data):
    df_data = pd.DataFrame(data)
    df_data = df_data.set_index('datetime')
    df_data.index = pd.to_datetime(df_data.index)
    return df_data


def datasets():
    df = add_basic_features(pd.read_hdf("../../datasets/bitfinex__BTC_USD.hdf", "table"))
    df['volume'] = df['volume_buy'] + df['volume_sell']

    datasets = [
        {
            'name': 'BTCUSD',
            'exchange': 'bitfinex',
            'bars': df,
            'addtitonal_datasets': {
                "self": dict(bars=df)
            },
            'consolidators': [
                dict(type='percent', kwargs=dict(threshold=0.005)),
            ],
            'date_range': {
                'start': '2010-01-01',
                'end': '2022-01-01',  # not including
            }
        },
    ]

    resp = consolidate_datasets(datasets)

    # resp[0]['bars']['signal'] = np.random.random(resp[0]['bars'].shape[0])
    resp[0]['bars']['signal'] = np.array(
        [0.302481352895025, 0.7609441758030615, 0.8955795700602514, 0.0570384878850736, 0.3321521768872242,
         0.2787970970751995, 0.0650503479618545, 0.08804615224891654, 0.7995282393375976, 0.6181353728382173,
         0.5350322960334062, 0.36929813938367695, 0.15046026161803538, 0.24673171130737404, 0.8166587500127691,
         0.11732518420562976, 0.8533731512325953, 0.9215320359385151, 0.08520434469624494, 0.005383298862784547,
         0.5204950528746524, 0.016428689272717012, 0.18016366491424984])

    resp[0]['bars']['exp_win'] = 0.005
    resp[0]['bars']['exp_loss'] = -0.006

    return resp


DATASETS = datasets()


@pytest.mark.parametrize(
    'datasets',
    (
        DATASETS,
    )
)
def test_Backtrader_consolidation_2nd_dataset(datasets):  # noqa
    df_data = datasets[0]['bars']
    df_res = datasets[0]['addtitonal_datasets']['self']['bars']

    print("ASDSDASDASDSADSDAS")
    print(list(df_data['signal'].values))

    assert (df_data.open.values == df_res.open.values).all()
    assert (df_data.high.values == df_res.high.values).all()
    assert (df_data.low.values == df_res.low.values).all()
    assert (df_data.close.values == df_res.close.values).all()
    assert (df_data.volume.values == df_res.volume.values).all()


@pytest.mark.parametrize(
    'datasets',
    (
        DATASETS,
    )
)
def test_Backtrader_Simulation_run(datasets):  # noqa
    df_data = datasets[0]['bars']

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

    assert df_sim.balance.values[-1] == 740793.2485289685


@pytest.mark.parametrize(
    'datasets',
    (
        DATASETS,
    )
)
def test_Backtrader_Simulation_run_limit(datasets):  # noqa
    df_data = datasets[0]['bars']

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
        use_worst_pnl=False,
        entry_limit=True
    )

    assert df_sim.balance.values[-1] == 745700.9576640638


@pytest.mark.parametrize(
    'datasets',
    (
        DATASETS,
    )
)
def test_Backtrader_Simulation_custom_sizing(datasets):  # noqa
    df_data = datasets[0]['bars']

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

    assert df_sim.balance.values[-1] == 801287.4259508352


@pytest.mark.parametrize(
    'datasets',
    (
        DATASETS,
    )
)
def test_Backtrader_Simulation_risk_management(datasets):  # noqa
    df_data = datasets[0]['bars']

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

    assert df_sim.balance.values[-1] == 740793.2485289685


@pytest.mark.parametrize(
    'datasets',
    (
        DATASETS,
    )
)
def test_Data_Feed_Factory(datasets):  # noqa
    df_data = datasets[0]['bars']

    with pytest.raises(DataFeedMissingFieldsException):
        bt_data_feed_factory(df_data[['close', 'open']])

    data_feed_cls = bt_data_feed_factory(df_data)

    assert set(['exp_win', 'exp_loss']) - set(data_feed_cls.datafields) == set()


@pytest.mark.parametrize(
    'datasets',
    (
        DATASETS,
    )
)
def test_Backtrader_Simulation_run_2nd_exchange_sanity_check(datasets):  # noqa
    """
    Base equity (e.g. predict and trade same exchange) should be the same as
    predict and trade same exchange but with wnd exchange interface
    :param data:
    :return:
    """
    df_data = datasets[0]['bars']
    df_data_2nd = datasets[0]['addtitonal_datasets']['self']['bars']

    exchange = Bitfinex(costs=Costs(
        taker_fee=-0.155 / 100,
        maker_fee=-0.2 / 100,
        drift=-0.0 / 100,
        impact=-0.1 / 100,
    ))

    bitfinex_fee = 0.155 / 100

    expected_profit = 0.015 - bitfinex_fee * 2
    expected_loss = 0.015 - bitfinex_fee * 2

    df_data_2nd['signal'] = df_data['signal']

    trades, df_sim = get_trading_simulation(
        df_data=df_data_2nd,
        sizing=KellySizing(expected_profit, expected_loss, 10, direction='both'),
        exchange=exchange,
        use_worst_pnl=False
    )

    assert df_sim.balance.values[-1] == 740793.2485289685
