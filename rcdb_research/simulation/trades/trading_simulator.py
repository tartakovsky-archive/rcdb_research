from __future__ import annotations

from typing import List, NamedTuple, Optional
import logging

import numpy as np
import pandas as pd
from .account import Account, AccountManager
from .exchange import Exchange, BidAsk
from .execution import MMMEA, ExecutionManager, ExecutionAlgo
from .portfolio import Portfolio
from .position import Position, PositionManager, PositionAction, PositionChange
from .sizing import SizingAlgo
from .trades import Trades


class Context(NamedTuple):
    proba: float
    bidask: BidAsk
    account_pre_trade: Account
    position_desired: Optional[Position]
    actions: List[PositionAction]
    changes: List[PositionChange]
    realized_pnl: float
    account_post_trade: Account

    def tell_me_a_story(self) -> str:
        position_pre_trade = self.account_pre_trade.portfolio.position
        position_post_trade = self.account_post_trade.portfolio.position
        change_strs = '\n            '.join([
            f'size={c.size:.4f}, '
            f'price=${c.avg_price:.3f}, '
            f'fee=${c.fee:.4f}, '
            f'slippage=${c.slippage:.3f}' for c in self.changes
        ])

        story = f"""----- New bar -----

        Bid = ${self.bidask.bid:.2f}, Ask = ${self.bidask.ask:.2f}
        
        Current position:
            size = {position_pre_trade.size if position_pre_trade is not None else 0:.2f}
            avg_price = ${position_pre_trade.avg_price if position_pre_trade is not None else 0:.3f}
        
        Current account:
            balance = ${self.account_pre_trade.balance:.2f}
            exposure = {self.account_pre_trade.metrics.exposure:.4f}
            unrealized PnL = ${self.account_pre_trade.metrics.pnl_after_fees:.2f} (after fees)
        
        Predicted proba = {self.proba}
        Desired position:
            size = {self.position_desired.size if self.position_desired is not None else 0:.2f}
            price = ${self.position_desired.avg_price if self.position_desired is not None else 0:.2f}

        Actions to be taken:
            {self.actions}

        {self.account_pre_trade.exchange.costs}

        Executed changes:
            {change_strs}

        Realized PnL = ${self.realized_pnl:.2f}

        New position:
            size = {position_post_trade.size if position_post_trade is not None else 0:.2f}
            avg_price = ${position_post_trade.avg_price if position_post_trade is not None else 0:.3f}

        New account:
            balance = ${self.account_post_trade.balance:.2f}
            exposure = {self.account_post_trade.metrics.exposure:.4f}
            unrealized_pnl_after_fees = ${self.account_post_trade.metrics.pnl_after_fees:.2f}
        """
        return story


class TradingSimulator:

    ############
    # Initialization
    ############
    def __init__(self,
                 exchange: Exchange,
                 sizing_algo: SizingAlgo,
                 execution_algo: ExecutionAlgo = MMMEA(),
                 initial_equity: float = 100,
                 compounded: bool = False):
        self.exchange = exchange
        self.sizing_algo = sizing_algo
        self.execution_algo = execution_algo
        self.initial_equity = initial_equity
        self.compounded = compounded

    def on_bar(self, proba: float, price: BidAsk, prior_contexts: list) -> Context:
        costs = self.exchange.costs

        # Mark account to market
        account_pre_trade = AccountManager.mark_to_market(
            account=prior_contexts[0].account_post_trade,
            price=price
        )
        position_pre_trade = account_pre_trade.portfolio.position

        # Calculate desired position
        # desired_portfolio = Rebalancer.rebalance_portfolio(proba, price, account) ?
        # TODO: replace sizing_fn with SizingAlgo interface
        # TODO: add non-compounding option?
        # TODO: size based on balance + unrealized pnl?
        desired_size = self.sizing_algo.size(proba) * account_pre_trade.balance
        desired_price = price.ask if desired_size > 0 else price.bid if desired_size < 0 else None
        desired_position = Position(desired_size, desired_price) if desired_size != 0 else None

        # Calculate diff between current and desired position
        desired_actions = PositionManager.diff(
            current_position=position_pre_trade,
            desired_position=desired_position
        )

        # Execute the diff
        executed_changes = ExecutionManager.execute(
            actions=desired_actions,
            price=price,
            costs=costs,
            algo=self.execution_algo
        )

        # Incorporate changes into the account
        account_post_trade, pnl_realized = AccountManager.merge_changes(
            account=account_pre_trade,
            changes=executed_changes
        )

        return Context(
            proba=proba,
            bidask=price,
            account_pre_trade=account_pre_trade,
            position_desired=desired_position,
            actions=desired_actions,
            changes=executed_changes,
            realized_pnl=pnl_realized,
            account_post_trade=account_post_trade
        )

    def _contexts(self, probas: np.ndarray, bidasks: List[BidAsk]) -> List[Context]:
        account_pre_trade = Account(
            exchange=self.exchange,
            balance=self.initial_equity,
            portfolio=Portfolio(position=None, price=bidasks[0])
        )

        initial_context = Context(
            proba=0.5,
            bidask=bidasks[0],
            account_pre_trade=account_pre_trade,
            position_desired=None,
            actions=[],
            changes=[],
            realized_pnl=0,
            account_post_trade=account_pre_trade
        )

        contexts = [initial_context]

        for i in range(probas.size):
            context = self.on_bar(proba=probas[i], price=bidasks[i], prior_contexts=list(reversed(contexts)))
            contexts.append(context)

        return contexts[1:]

    def trades(self, probas: np.ndarray, data: pd.DataFrame) -> Trades:
        # Check that data has required columns
        required_columns = ['bid', 'ask']  # ['open', 'high', 'low', 'close', 'bid', 'ask']
        missing_columns = [c for c in required_columns if c not in data.columns]
        if len(missing_columns) > 0:
            raise ValueError(
                f'\ndata.columns should contain {required_columns}\n'
                f'Columns {missing_columns} are missing'
            )

        if probas.size > data.index.size:
            raise ValueError(f'probas.size={probas.size} should be >= data.index.size={data.index.size}')

        if data.index.size > probas.size:
            logging.warning(
                f' Last {probas.size} out of {data.index.size} '
                'elements will be taken from data to match probas size'
            )

        df = data[-probas.size:]

        bidasks = [BidAsk(bid, ask) for bid, ask in zip(df.bid.values, df.ask.values)]

        contexts = self._contexts(probas=probas, bidasks=bidasks)

        balance = np.array([c.account_pre_trade.balance for c in contexts])
        exposure = np.array([c.account_pre_trade.metrics.exposure for c in contexts])
        unrealized_pnl = np.array([c.account_pre_trade.metrics.pnl_after_fees for c in contexts])

        return Trades(
            balance=balance,
            exposure=exposure,
            unrealized_pnl=unrealized_pnl,
            context=contexts,
            index=df.index
        )
