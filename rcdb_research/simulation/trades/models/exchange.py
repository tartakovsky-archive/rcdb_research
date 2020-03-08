from typing import NamedTuple, List

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
    pair: Pair
    bid: float
    ask: float

    @property
    def mid_market(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class Pairs(Enum):
    btcusd = Pair(base=Currency(name='Bitcoin', symbol='BTC'), quote=Currency(name='United States Dollar', symbol='USD'))
    xbtusd = Pair(base=Currency(name='Bitcoin', symbol='XBT'), quote=Currency(name='United States Dollar', symbol='USD'))
    ethusd = Pair(base=Currency(name='Ethereum', symbol='ETH'), quote=Currency(name='United States Dollar', symbol='USD'))
    ethbtc = Pair(base=Currency(name='Ethereum', symbol='ETH'), quote=Currency(name='Bitcoin', symbol='BTC'))


class Costs(NamedTuple):
    taker_fee: float
    maker_fee: float
    drift: float
    impact: float

    @property
    def slippage(self) -> float:
        return self.drift + self.impact


class Exchange(NamedTuple):
    name: str
    initial_margin: float
    maintenance_margin: float
    costs: Costs
    pairs: List[Pair]

    @property
    def max_leverage(self) -> float:
        return 1 / self.initial_margin

    @property
    def liquidation_leverage(self) -> float:
        return 1 / self.maintenance_margin


class SupportedExchange(Enum):
    bitfinex = Exchange(
        name='bitfinex',
        costs=Costs(
            taker_fee=-0.2 / 100,
            maker_fee=-0.2 / 100,
            drift=-0.01 / 100,
            impact=-0.01 / 100,
        ),
        initial_margin=0.16666,
        maintenance_margin=0.15,
        pairs=[Pairs.btcusd.value, Pairs.ethusd.value, Pairs.ethbtc.value]
    )
    bitmex = Exchange(
        name='bitmex',
        costs=Costs(
            taker_fee=-0.075 / 100,
            maker_fee=0.025 / 100,
            drift=-0.1 / 100,
            impact=-0.1 / 100,
        ),
        initial_margin=0.01,
        maintenance_margin=0.005,
        pairs=[Pairs.xbtusd.value]
    )
