from __future__ import annotations

from dataclasses import dataclass

import weakref

from typing import NamedTuple, List, Tuple

from .exchange import Exchange, BidAsk
from .portfolio import Portfolio, PortfolioManager
from .position import PositionManager, PositionChange


@dataclass
class Account:
    exchange: Exchange
    balance: float
    portfolio: Portfolio

    def __post_init__(self):
        self.metrics = AccountMetrics(self)


class AccountMetrics:
    def __init__(self, account: Account):
        self.account = weakref.proxy(account)

    @property
    def pnl_unrealized(self) -> float:
        return self.account.portfolio.metrics.pnl_unrealized

    @property
    def pnl_after_fees(self) -> float:
        return self.account.portfolio.metrics.pnl_after_fees(self.account.exchange.costs)

    @property
    def exposure(self) -> float:
        position = self.account.portfolio.position
        if position is None:
            return 0

        return position.size / self.account.balance


class AccountManager(NamedTuple):
    @staticmethod
    def mark_to_market(account: Account, price: BidAsk) -> Account:
        return Account(
            account.exchange,
            account.balance,
            PortfolioManager.mark_to_market(account.portfolio, price)
        )

    @staticmethod
    def merge_change(account: Account, change: PositionChange) -> Tuple[Account, float]:
        position = account.portfolio.position

        new_position, pnl_realized = PositionManager.merge_change(position, change)
        price = BidAsk(change.avg_price, change.avg_price)

        new_account = Account(
            exchange=account.exchange,
            balance=account.balance + pnl_realized,
            portfolio=Portfolio(position=new_position, price=price)
        )

        return new_account, pnl_realized

    @staticmethod
    def merge_changes(account: Account, changes: List[PositionChange]) -> Tuple[Account, float]:
        new_account = account
        pnl = 0

        for change in changes:
            new_account, pnl_realized = AccountManager.merge_change(new_account, change)
            pnl += pnl_realized

        return new_account, pnl
