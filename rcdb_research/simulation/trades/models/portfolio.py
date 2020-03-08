from __future__ import annotations

from dataclasses import dataclass

import weakref

from copy import deepcopy
from typing import NamedTuple, Dict, Optional, List

from .exchange import BidAsk, Pair, Costs
from .position import Position, PositionAction, PositionManager


@dataclass
class Portfolio:
    entries: Dict[Pair, PortfolioEntry]

    def __post_init__(self):
        self.metrics = PortfolioMetrics(self)


class PortfolioEntry(NamedTuple):
    position: Position
    price: BidAsk


class PortfolioMetrics:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = weakref.proxy(portfolio)

    @property
    def unrealized_portfolio_pnl(self) -> float:
        return sum(self.unrealized_position_pnls.values())

    @property
    def unrealized_position_pnls(self) -> Dict[Pair, float]:
        def entry_pnl(entry: PortfolioEntry) -> float:
            direction = entry.position.direction
            mark_price = entry.price.bid if direction == 1 else entry.price.ask
            pnl_pct = direction * (mark_price / entry.position.avg_price - 1)
            pnl = pnl_pct * abs(entry.position.size)
            return pnl

        pnls = {pair: entry_pnl(entry) for pair, entry in self.portfolio.entries.items()}

        return pnls

    def after_fees_portfolio_pnl(self, costs: Costs) -> float:
        return sum(self.after_fees_position_pnls(costs).values())

    def after_fees_position_pnls(self, costs: Costs) -> Dict[Pair, float]:
        def entry_pnl(entry: PortfolioEntry) -> float:
            direction = entry.position.direction
            # Long position would be closed at bid, price after slippage would become lower
            # Short position would be closed at ask, price after slippage would become higher
            mark_price = entry.price.bid if direction == 1 else entry.price.ask
            slipped_price = mark_price * (1 - direction * abs(costs.slippage))
            pnl_pct = direction * (slipped_price / entry.position.avg_price - 1)
            pnl = pnl_pct * abs(entry.position.size)
            pnl_after_fees = pnl + abs(entry.position.size) * costs.taker_fee
            return pnl_after_fees

        pnls = {pair: entry_pnl(entry) for pair, entry in self.portfolio.entries.items()}

        return pnls


class PortfolioManager:
    @staticmethod
    def update(portfolio: Portfolio, pair: Pair, entry: Optional[PortfolioEntry]) -> Portfolio:
        new_entries = deepcopy(portfolio.entries)

        if entry is None:
            _ = new_entries.pop(pair, None)
        else:
            new_entries[pair] = entry

        return Portfolio(entries=new_entries)

    @staticmethod
    def mark_to_market(portfolio: Portfolio, pair: Pair, price: BidAsk) -> Portfolio:
        if pair in portfolio.entries:
            new_entries = deepcopy(portfolio.entries)
            new_entries[pair] = new_entries[pair]._replace(price=price)

            return Portfolio(entries=new_entries)

        return portfolio

    @staticmethod
    def diff(current_portfolio: Portfolio, desired_portfolio: Portfolio) -> Dict[
        Pair,
        List[PositionAction]
    ]:
        pairs_to_close = [p for p in current_portfolio.entries.keys() if p not in desired_portfolio.entries.keys()]
        pairs_to_open = [p for p in desired_portfolio.entries.keys() if p not in current_portfolio.entries.keys()]
        pairs_to_adjust = [p for p in desired_portfolio.entries.keys() if p in current_portfolio.entries.keys()]

        close_actions = [PositionManager.diff(pair, current_portfolio.entries[pair].position, None)
                         for pair in pairs_to_close]

        open_actions = [PositionManager.diff(pair, None, desired_portfolio.entries[pair].position)
                        for pair in pairs_to_open]

        adjust_actions = [PositionManager.diff(pair, current_portfolio.entries[pair].position,
                                               desired_portfolio.entries[pair].position)
                          for pair in pairs_to_adjust]

        actions: List[dict] = close_actions + open_actions + adjust_actions
        actions: dict = {k: v for d in actions for k, v in d.items()}

        return actions
