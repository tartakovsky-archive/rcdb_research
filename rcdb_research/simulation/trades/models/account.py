from __future__ import annotations

from dataclasses import dataclass

import weakref

from typing import NamedTuple, Dict, List

from .exchange import Exchange, Pair, BidAsk
from .execution import ExecutionResult
from .portfolio import Portfolio


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
    def unrealized_pnl(self) -> float:
        return self.account.portfolio.unrealized_portfolio_pnl

    @property
    def after_fees_pnl(self) -> float:
        return self.account.portfolio.after_fees_portfolio_pnl(self.account.exchange.costs)

    @property
    def long_exposure(self) -> float:
        exposures = [entry.position.size / self.account.balance for entry in self.account.portfolio.entries.values()]
        return sum([exp for exp in exposures if exp > 0])

    @property
    def short_exposure(self) -> float:
        exposures = [entry.position.size / self.account.balance for entry in self.account.portfolio.entries.values()]
        return sum([exp for exp in exposures if exp < 0])

    @property
    def net_exposure(self) -> float:
        exposures = [entry.position.size / self.account.balance for entry in self.account.portfolio.entries.values()]
        return sum([exp for exp in exposures])

    @property
    def gross_exposure(self) -> float:
        exposures = [entry.position.size / self.account.balance for entry in self.account.portfolio.entries.values()]
        return sum([abs(exp) for exp in exposures])


class AccountManager(NamedTuple):
    @staticmethod
    def mark_to_market(account: Account, prices: Dict[Pair, BidAsk]) -> Account:
        new_portfolio = account.portfolio
        for pair, price in prices.items():
            new_portfolio = new_portfolio.mark_to_market(pair, price)

        return Account(account.exchange, account.balance, new_portfolio)

    @staticmethod
    def merge_changes(account, changes: Dict[Pair, List[ExecutionResult]]) -> Account:
        return account
