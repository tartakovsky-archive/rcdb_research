from __future__ import annotations

from dataclasses import dataclass

import weakref

from typing import NamedTuple, Optional

from .exchange import BidAsk, Costs
from .position import Position


@dataclass
class Portfolio:
    entry: Optional[PortfolioEntry]

    def __post_init__(self):
        self.metrics = PortfolioMetrics(self)


class PortfolioEntry(NamedTuple):
    position: Position
    price: BidAsk


class PortfolioMetrics:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = weakref.proxy(portfolio)

    @property
    def pnl_unrealized(self) -> float:
        entry = self.portfolio.entry

        if entry is None:
            return 0

        direction = entry.position.direction
        mark_price = entry.price.bid if direction == 1 else entry.price.ask
        pnl_pct = direction * (mark_price / entry.position.avg_price - 1)
        pnl = pnl_pct * abs(entry.position.size)
        return pnl

    def pnl_after_fees(self, costs: Costs) -> float:
        entry = self.portfolio.entry

        if entry is None:
            return 0

        direction = entry.position.direction
        # Long position would be closed at bid, price after slippage would become lower
        # Short position would be closed at ask, price after slippage would become higher
        mark_price = entry.price.bid if direction == 1 else entry.price.ask
        slipped_price = mark_price * (1 - direction * abs(costs.slippage))
        pnl_pct = direction * (slipped_price / entry.position.avg_price - 1)
        pnl = pnl_pct * abs(entry.position.size)
        pnl_after_fees = pnl + abs(entry.position.size) * costs.taker_fee

        return pnl_after_fees


class PortfolioManager:
    @staticmethod
    def mark_to_market(portfolio: Portfolio, price: BidAsk) -> Portfolio:
        return Portfolio(entry=PortfolioEntry(position=portfolio.entry.position, price=price))
