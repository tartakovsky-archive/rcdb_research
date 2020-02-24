from __future__ import annotations

from typing import NamedTuple, Optional
from enum import Enum

import pandas as pd


class Exchange(NamedTuple):
    name: str


class Currency(NamedTuple):
    name: str
    symbol: str


class Pair(NamedTuple):
    base: Currency
    quote: Currency

    def __str__(self):
        return f"{self.base.symbol}/{self.quote.symbol}"


class Fee(NamedTuple):
    taker: float
    maker: float

    @staticmethod
    def for_exchange(exchange: SupportedExchange) -> Fee:
        if exchange is SupportedExchange.bitfinex:
            return Fee(taker=-0.2 / 100, maker=-0.2 / 100)
        elif exchange is SupportedExchange.bitmex:
            return Fee(taker=-0.075 / 100, maker=0.025 / 100)
        elif exchange is SupportedExchange.binance:
            return Fee(taker=-0.075 / 100, maker=-0.075 / 100)


class EquityState(NamedTuple):
    quote_holdings: float
    base_holdings: float
    base_holdings_quote_value: float

    @property
    def equity(self) -> float:
        return self.quote_holdings + self.base_holdings_quote_value

    @property
    def exposure(self) -> float:
        return self.base_holdings_quote_value / self.equity

    def __str__(self):
        params = str(self.to_dict()).strip("{}").replace(': ', '=').replace("'", '')
        return f"{self.__class__.__name__}({params})"

    def to_dict(self, prefix=""):
        d = dict(self._asdict())
        d.update({'equity': self.equity, 'exposure': self.exposure})
        d = {prefix + k: v for (k, v) in d.items()}
        return d


class OrderType(Enum):
    market = 'market'
    limit = 'limit'


class Order(NamedTuple):
    type: OrderType
    exchange: SupportedExchange
    pair: Pair
    price: float
    size: float

    def __str__(self):
        params = str(self.to_dict()).strip("{}").replace(': ', '=').replace("'", '')
        return f"{self.__class__.__name__}({params})"

    def to_dict(self, prefix=""):
        return {
            f"{prefix}type": self.type.value,
            f"{prefix}exchange": self.exchange.value.name,
            f"{prefix}pair": str(self.pair),
            f"{prefix}price": self.price,
            f"{prefix}size": self.size
        }


class TradeSignal(NamedTuple):
    proba: float
    desired_price: float
    desired_exposure: float

    def __str__(self):
        params = str(self.to_dict()).strip("{}").replace(': ', '=').replace("'", '')
        return f"{self.__class__.__name__}({params})"

    def to_dict(self, prefix=""):
        d = dict(self._asdict())
        d = {prefix + k: v for (k, v) in d.items()}
        return d


class ExecutionResult(NamedTuple):
    order: Order
    avg_price: float
    base_change: float
    quote_change: float

    def __str__(self):
        order_params = str(self.order.to_dict()).strip("{}").replace(': ', '=').replace("'", '')
        order_str = f"{self.order.__class__.__name__}({order_params})"
        d = self.to_dict()
        d['order'] = order_str
        params = str(d).strip("{}").replace(': ', '=').replace("'", '')
        return f"{self.__class__.__name__}({params})"

    def to_dict(self, prefix=""):
        d = dict(self._asdict())
        d['order'] = d['order'].to_dict()
        d = {prefix + k: v for (k, v) in d.items()}
        return d


class BarContext(NamedTuple):
    pre_trade_state: EquityState
    trade_signal: Optional[TradeSignal]
    order: Optional[Order]
    execution_result: Optional[ExecutionResult]
    post_trade_state: EquityState

    def __str__(self):
        order_params = str(self.order.to_dict()).strip("{}").replace(': ', '=').replace("'", '')
        order_str = f"{self.order.__class__.__name__}({order_params})"
        d = self.to_dict()
        d['order'] = order_str
        params = str(d).strip("{}").replace(': ', '=').replace("'", '')
        return f"{self.__class__.__name__}({params})"

    def to_dict(self, prefix=""):
        d = dict(self._asdict())

        d = {prefix + k: v.to_dict() for (k, v) in d.items() if v is not None}

        return d

    def to_df(self):
        d = dict(self._asdict())

        nd = {}
        nd.update(d['pre_trade_state'].to_dict(prefix="pre_"))

        trade_signal = d.get('trade_signal', None)
        if trade_signal is not None:
            nd.update(trade_signal.to_dict())

        order = d.get('order', None)
        if order is not None:
            nd.update({(f"order_{k}" if k == 'type' else k): v for (k, v) in order.to_dict().items()})

        execution_result = d.get('execution_result', None)
        if execution_result is not None:
            nd.update({k: v for (k, v) in execution_result.to_dict().items() if k != 'order'})

        nd.update(d['post_trade_state'].to_dict(prefix="post_"))

        return pd.DataFrame([nd])


# Enums

class SupportedExchange(Enum):
    bitfinex = Exchange(name='bitfinex')
    bitmex = Exchange(name='bitmex')
    binance = Exchange(name='binance')


class Pairs(Enum):
    btcusd = Pair(base=Currency(name='Bitcoin', symbol='BTC'), quote=Currency(name='United States Dollar', symbol='USD'))
    xbtusd = Pair(base=Currency(name='Bitcoin', symbol='XBT'), quote=Currency(name='United States Dollar', symbol='USD'))
    ethusd = Pair(base=Currency(name='Ethereum', symbol='ETH'), quote=Currency(name='United States Dollar', symbol='USD'))
    ethbtc = Pair(base=Currency(name='Ethereum', symbol='ETH'), quote=Currency(name='Bitcoin', symbol='BTC'))
