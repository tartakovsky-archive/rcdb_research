from __future__ import annotations

from dataclasses import dataclass

import weakref

from typing import NamedTuple, List

from .exchange import Exchange, BidAsk
from .execution import ExecutionResult
from .portfolio import Portfolio, PortfolioManager, PortfolioEntry
from .position import PositionManager


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
        return self.account.portfolio.pnl_unrealized

    @property
    def pnl_after_fees(self) -> float:
        return self.account.portfolio.pnl_after_fees(self.account.exchange.costs)

    @property
    def exposure(self) -> float:
        return self.account.portfolio.entry.position.size / self.account.balance


class AccountManager(NamedTuple):
    @staticmethod
    def mark_to_market(account: Account, price: BidAsk) -> Account:
        return Account(
            account.exchange,
            account.balance,
            PortfolioManager.mark_to_market(account.portfolio, price)
        )

    @staticmethod
    def merge_change(account: Account, change: ExecutionResult) -> Account:
        entry = account.portfolio.entry

        new_position, pnl_realized = PositionManager.merge_change(entry.position, change)

        new_entry = PortfolioEntry(
            new_position,
            BidAsk(
                change.avg_price * (1 - account.exchange.costs.spread / 2),
                change.avg_price * (1 + account.exchange.costs.spread / 2)
            )
        ) if new_position is not None else None

        return Account(
            exchange=account.exchange,
            balance=account.balance + pnl_realized,
            portfolio=Portfolio(entry=new_entry)
        )

    @staticmethod
    def merge_changes(account: Account, changes: List[ExecutionResult]) -> Account:
        new_account = account
        for change in changes:
            new_account = AccountManager.merge_change(new_account, change)

        return new_account
