from types import SimpleNamespace as sn

import pytest
import numpy as np
import pandas as pd

from .test_cross_validators import cv_result  # noqa
from rcdb_research.cross_validators import CVResult
from rcdb_research.trading_simulation import TradingSimulation


def raises(func, exception_msg=None, exception=ValueError):
    with pytest.raises(exception) as exc:
        func()

    if exception_msg:
        assert str(exc.value) == exception_msg

    return exc


def test_TradingSimulation_init_different_indexes(cv_result: CVResult):  # noqa
    raises(
        lambda: TradingSimulation(
            cvres=cv_result,
            y_df=pd.DataFrame([], index=cv_result.y_true.index * 0.5)
        ),
        'cvres.y_true.index is not equal to y_df.index'
    )


@pytest.mark.parametrize('position_size', [-.1, 1.1])  # noqa
def test_TradingSimulation_init_wrong_position_size(cv_result: CVResult, position_size):
    raises(
        lambda: TradingSimulation(
            cvres=cv_result,
            y_df=pd.DataFrame([], index=cv_result.y_true.index),
            position_size=position_size
        ),
        'position_size should be 0 < position_size <= 1. It is a fraction of initial_equity to trade'
    )


def test_TradingSimulation_init_wrong_order_type(cv_result: CVResult):  # noqa
    raises(
        lambda: TradingSimulation(
            cvres=cv_result,
            y_df=pd.DataFrame([], index=cv_result.y_true.index),
            entry_order='x'
        ),
        "entry_order=x: unknown order. Should be one of the following: ['market', 'limit']"
    )


REQUIERD_YDF_COLUMNS = ('open', 'high', 'low', 'change')
missing_columns_params = [
    (REQUIERD_YDF_COLUMNS[:i], REQUIERD_YDF_COLUMNS[i:]) for i in range(len(REQUIERD_YDF_COLUMNS))
]
missing_columns_params += [
    ([x for x in REQUIERD_YDF_COLUMNS if x != REQUIERD_YDF_COLUMNS[i]], REQUIERD_YDF_COLUMNS[i])
    for i in range(len(REQUIERD_YDF_COLUMNS))
]


@pytest.mark.parametrize(  # noqa
    'columns, missing_columns',
    missing_columns_params,
    ids=[f'missing: {" ".join(x[1])}' for x in missing_columns_params]
)
def test_TradingSimulation_init_wrong_y_df_columns(cv_result: CVResult, columns, missing_columns):
    exc = raises(
        lambda: TradingSimulation(
            cvres=cv_result,
            y_df=pd.DataFrame([], index=cv_result.y_true.index, columns=columns),
        )
    )
    assert 'y_df is missing required columns:' in str(exc.value)
    for mc in missing_columns:
        assert mc in str(exc.value)


@pytest.fixture(
    params=[
        (
            dict(
                no_reentry=False,
                entry_order='limit',
            ),
            pd.Series([
                -0.075, 0, 0, 0, -0.075, -0.075, 0, 0, 0, -0.075, 0, -0.075, 0, 0, -0.075, 0, 0, 0, -0.075, -0.075
            ])
        ),
        (
            dict(
                no_reentry=False,
                entry_order='market',
            ),
            pd.Series([
                -0.2, 0, 0, -0.2, -0.2, -0.2, 0, -0.2, -0.2, -0.2, -0.2, -0.2, 0, -0.2, -0.2, -0.2, -0.2, 0, -0.2, -0.2
            ])
        ),
        (
            dict(
                no_reentry=True,
                entry_order='market',
            ),
            pd.Series([
                -0.2, 0, 0, -0.1, 0, -0.1, 0, -0.1, 0, 0, 0, -0.1, 0, -0.1, 0, 0, -0.1, 0, -0.1, -0.1
            ])
        ),
        (
            dict(
                no_reentry=True,
                entry_order='limit',
            ),
            pd.Series([
                -0.075, 0, 0, 0, 0.025, -0.1, 0, 0, 0, 0.025, 0, -0.1, 0, 0, .025, 0, -0.1, 0, 0.025, -0.1
            ])
        ),
    ]
)
def fee_params_and_res(request):
    additional_params, test_fee = request.param
    params = dict(
        cvres=sn(
            y_pred=pd.Series([1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]),
            y_true=pd.Series([1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]),
        ),
        y_df=pd.DataFrame(
            dict(
                open=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                low=[9, 10, 10, 10, 9, 9, 10, 10, 10, 9, 10, 9, 10, 10, 9, 10, 10, 10, 9, 9],
                high=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                change=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            )
        ),

        maker_fee=0.025,
        taker_fee=-0.1
    )
    params.update(additional_params)
    return params, test_fee


def test_TradingSimulation_init_fee(fee_params_and_res):
    params, test_fee = fee_params_and_res
    assert (np.abs(TradingSimulation(**params).fees - test_fee) < 10e-15).all()


@pytest.fixture(params=[True, False])  # noqa
def trading_simulation(request):  # noqa
    y_pred = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_true = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1])
    if request.param:  # dt index
        cvresult = CVResult(
            y_pred=y_pred,
            y_true=y_true,
            index=pd.date_range('12-11-2019', periods=y_pred.size)
        )
    else:
        cvresult = CVResult(
            y_pred=y_pred,
            y_true=y_true,
        )

    y_df = pd.DataFrame(
        dict(
            open=[10, 10, 10, 9, 10, 10, 10, 10, 9, 10],
            low=[9, 10, 10, 10, 9, 9, 10, 10, 10, 9],
            high=[10, 12, 13, 12, 10, 10, 10, 11, 12, 10],
            change=[0, 2, 1, -1, -2, 0, 0, 0, 1, 1]
        ),
        index=cvresult.y_true.index
    )

    return TradingSimulation(cvres=cvresult, y_df=y_df)


@pytest.mark.parametrize(
    "tradingsimulator_attr, test_cvresult_attr, test_result_values",
    (
        ('wins', 'tp', np.array([0.0005, -0.9995, -1.9995, 0.0005, 1.0005])),
        ('losses', 'fp', np.array([0.0005])),
        ('returns', None, np.array([0.0005, 0.0, 0.0, -0.9995, -1.9995, 0.0005, 0.0, 0.0005, 0.0, 1.0005])),
        ('equity', None, np.array(
            [100.0, 100.0, 100.0, 50.025, -49.95000000000001, -49.95000000000001, -49.95000000000001,
             -49.95000000000001, -49.95000000000001, -49.95000000000001]
        )),
        ('cum_return', None, np.array([0.0, 0.0, 0.0, -0.49975, -1.4995, -1.4995, -1.4995, -1.4995, -1.4995, -1.4995])),
        ('drawdown', None, np.array(
            [0.0, 0.0, 0.0, -0.49975, -1.4995, -1.4995, -1.4995, -1.4995, -1.4995, -1.4995]
        )),
        (
            'expected_cum_return',
            None,
            np.array(
                [-0.0, -0.11886904761904761, -0.23773809523809522, -0.35660714285714284,
                 -0.47547619047619044, -0.5943452380952381, -0.7132142857142857,
                 -0.8320833333333333, -0.9509523809523809, -1.0698214285714285]
            )
        ),

    )
)
def test_TradingSimulation_metrics(
    trading_simulation: TradingSimulation,
    tradingsimulator_attr,
    test_cvresult_attr,
    test_result_values
):
    cvresult = trading_simulation.cvres

    if test_cvresult_attr:
        test_index = cvresult.y_true.index[getattr(cvresult, test_cvresult_attr)() == 1]
    else:
        test_index = cvresult.y_true.index

    res = getattr(trading_simulation, tradingsimulator_attr)
    assert np.array_equal(res.index, test_index)
    print(res.values.tolist())
    print(test_result_values)
    assert np.array_equal(res.values, test_result_values)


@pytest.mark.parametrize(
    "metrics_name, test_result",
    (
        ('mean_profit', -0.39949999999999997),
        ('mean_loss', 0.0005),
        ('expectancy', -0.3328333333333333),
        ('expectancy_times_recall', -0.23773809523809522),
    )
)
def test_TradingSimulation_metrics_scalars(trading_simulation, metrics_name, test_result):
    assert getattr(trading_simulation, metrics_name) == test_result
