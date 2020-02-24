from __future__ import annotations

from typing import NamedTuple, Optional
from enum import Enum


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


class OrderType(Enum):
    market = 'market'
    limit = 'limit'


class Order(NamedTuple):
    type: OrderType
    exchange: SupportedExchange
    pair: Pair
    price: float
    size: float


class TradeSignal(NamedTuple):
    proba: float
    desired_price: float
    desired_exposure: float


class ExecutionResult(NamedTuple):
    order: Order
    avg_price: float
    base_change: float
    quote_change: float


class BarContext(NamedTuple):
    pre_trade_state: EquityState
    trade_signal: Optional[TradeSignal]
    order: Optional[Order]
    execution_result: Optional[ExecutionResult]
    post_trade_state: EquityState


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
