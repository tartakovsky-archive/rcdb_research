from dataclasses import dataclass
from typing import NamedTuple

from enum import Enum


class Currency(NamedTuple):
    name: str
    symbol: str


class Pair(NamedTuple):
    base: Currency
    quote: Currency

    def __str__(self):
        return f"{self.base.symbol}/{self.quote.symbol}"


class BidAsk(NamedTuple):
    bid: float
    ask: float

    @property
    def mid_market(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class Pairs(Enum):
    btcusd = Pair(
        base=Currency(name='Bitcoin', symbol='BTC'),
        quote=Currency(name='United States Dollar', symbol='USD')
    )
    xbtusd = Pair(
        base=Currency(name='Bitcoin', symbol='XBT'),
        quote=Currency(name='United States Dollar', symbol='USD')
    )
    ethusd = Pair(
        base=Currency(name='Ethereum', symbol='ETH'),
        quote=Currency(name='United States Dollar', symbol='USD')
    )
    ethbtc = Pair(
        base=Currency(name='Ethereum', symbol='ETH'),
        quote=Currency(name='Bitcoin', symbol='BTC')
    )


class Costs(NamedTuple):
    taker_fee: float
    maker_fee: float
    drift: float
    impact: float

    @property
    def slippage(self) -> float:
        return self.drift + self.impact


@dataclass
class Exchange:
    name: str
    initial_margin: float
    maintenance_margin: float
    costs: Costs

    @property
    def max_leverage(self) -> float:
        return 1 / self.initial_margin

    @property
    def liquidation_leverage(self) -> float:
        return 1 / self.maintenance_margin


@dataclass
class NoFeex(Exchange):
    name: str = 'nofeex'
    initial_margin: float = 0.01
    maintenance_margin: float = 0.005
    costs: Costs = Costs(taker_fee=0, maker_fee=0, drift=0, impact=0)


@dataclass
class Bitfinex(Exchange):
    name: str = 'bitfinex'
    initial_margin: float = 0.16666
    maintenance_margin: float = 0.15
    costs: Costs = Costs(
        taker_fee=-0.2 / 100,
        maker_fee=-0.2 / 100,
        drift=-0.025 / 100,
        impact=-0.2 / 100,
    )


@dataclass
class Bitmex(Exchange):
    name: str = 'bitmex'
    initial_margin: float = 0.01
    maintenance_margin: float = 0.005
    costs: Costs = Costs(
        taker_fee=-0.075 / 100,
        maker_fee=0.025 / 100,
        drift=-0.01 / 100,
        impact=-0.01 / 100,
    )
