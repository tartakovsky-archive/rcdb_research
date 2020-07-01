import pytest
import pandas as pd

from rcdb_research.simulation import TradingSimulatorOld
from .test_predictions import predictions  # noqa


def raises(func, exception_msg=None, exception=ValueError):
    with pytest.raises(exception) as exc:
        func()

    if exception_msg:
        assert str(exc.value) == exception_msg

    return exc


REQUIERD_YDF_COLUMNS = ('open', 'high', 'low', 'close')
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
def test_TradingSimulator_init_wrong_ohlc_columns(predictions, columns, missing_columns):  # noqa
    exc = raises(
        lambda: TradingSimulatorOld().trades(
            predicts=predictions,
            ohlc=pd.DataFrame([], index=predictions.index, columns=columns),
        )
    )
    assert 'ohlc is missing required columns:' in str(exc.value)
    for mc in missing_columns:
        assert mc in str(exc.value)


def test_TradingSimulator_init_wrong_order_type():  # noqa
    raises(
        lambda: TradingSimulatorOld(entry_order='x'),
        "entry_order=x: unknown order. Should be one of the following: ['market', 'limit']"
    )


@pytest.mark.parametrize(
    'exchange, is_raises',
    (
        ('bitmex', False),
        ('bitfinex', False),
        ('binance', False),
        ('x', True),
    )
)
def test_TradingSimulator_init_wrong_exchange(exchange, is_raises):  # noqa
    f = lambda: TradingSimulatorOld(exchange=exchange)
    if is_raises:
        raises(
            f,
            f"exchange={exchange}: unknown exchange. Should be one of the following: ['bitmex', 'bitfinex', 'binance']"
        )
    else:
        f()
