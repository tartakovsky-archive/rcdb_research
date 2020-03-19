from __future__ import annotations

from dataclasses import dataclass

import weakref

from typing import Optional

from .exchange import BidAsk, Costs
from .position import Position


@dataclass
class Portfolio:
    position: Optional[Position]
    price: BidAsk

    def __post_init__(self):
        self.metrics = PortfolioMetrics(self)


class PortfolioMetrics:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = weakref.proxy(portfolio)

    @property
    def pnl_unrealized(self) -> float:
        position = self.portfolio.position
        price = self.portfolio.price

        if position is None:
            return 0

        direction = position.direction
        mark_price = price.bid if direction == 1 else price.ask
        pnl_pct = direction * (mark_price / position.avg_price - 1)
        pnl = pnl_pct * abs(position.size)
        return pnl

    def pnl_after_fees(self, costs: Costs) -> float:
        position = self.portfolio.position
        price = self.portfolio.price

        if position is None:
            return 0

        close_direction = -1 * position.direction  # position will be closed with reverse direction
        # Long position would be closed at bid, price after slippage would become lower
        # Short position would be closed at ask, price after slippage would become higher
        mark_price = price.ask if close_direction == 1 else price.bid
        slipped_price = mark_price * (1 + close_direction * abs(costs.slippage))

        price_change_pct = (slipped_price / position.avg_price - 1)
        pnl_pct = position.direction * price_change_pct
        pnl_abs = abs(position.size) * pnl_pct

        fee = abs(position.size) * costs.taker_fee
        pnl_after_fees = pnl_abs + fee

        return pnl_after_fees


class PortfolioManager:
    @staticmethod
    def mark_to_market(portfolio: Portfolio, price: BidAsk) -> Portfolio:
        return Portfolio(position=portfolio.position, price=price)
