from __future__ import annotations

from typing import Callable, List, Dict

import numpy as np
# import pandas as pd

# from rcdb_research.simulation import Probabilities, Trades

from .models import Exchange, SupportedExchange, Pair, BidAsk, PortfolioEntry
from .models import Account, Position, OrderType, Order, ExecutionResult, BarContext
from .execution.differ import Differ
from .execution.algos import ExecutionAlgo, MMMEA, ExecutionResult
from .execution.account import AccountManager

from typing import Callable, List


class TradingSimulator:

    ############
    # Initialization
    ############
    def __init__(self,
                 exchange: Exchange,
                 pair: Pair,
                 initial_equity: float,
                 sizing_fn: Callable,
                 compounded: bool = False):
        self.exchange = exchange
        self.pair = pair
        self.initial_equity = initial_equity
        self.sizing_fn = sizing_fn
        self.compounded = compounded

    def on_bar(self, proba: float, price: BidAsk, prior_contexts: list) -> tuple:
        costs = self.exchange.costs

        context_prior = prior_contexts[0]
        account_prior: Account = context_prior.account

        account_pre_trade = account_prior.mark_to_market({self.pair: price})

        # desired_portfolio = Rebalancer.rebalance_portfolio(proba, price, account)

        # TODO: replace sizing_fn with SizingAlgo interface
        # TODO: add non-compounding option?
        # TODO: size based on balance + unrealized pnl?
        desired_size = self.sizing_fn(proba) * account_pre_trade.balance
        desired_price = price.ask if desired_size > 0 else price.bid if desired_size < 0 else None
        desired_position = Position(price.pair, desired_size, desired_price) if desired_size != 0 else None
        desired_portfolio = account_pre_trade.portfolio.update(
            pair=self.pair,
            entry=PortfolioEntry(desired_position, price) if desired_position is not None else None
        )

        porfolio_diff = Differ.diff_portfolios(account_pre_trade.portfolio, desired_portfolio)

        # portfolio_changes = Execution.execute(porfolio_diff, algo)

        # TODO: Replace with ExecutionManager call?
        changes: Dict[Pair, List[ExecutionResult]] = {
            k: [MMMEA.execute(action, price, costs) for action in actions]
            for (k, actions) in porfolio_diff.items()
        }

        # TODO: Implement merge_changes method
        account_post_trade = AccountManager.merge_changes(account_pre_trade, changes)


def _on_proba(self, proba: float, bid: float, ask: float, prior_contexts: List[BarContext]) -> BarContext:
    # Typically used modifiers:
    # prior - last available value of the previous bar
    # pre_trade - current value before new trade is executed
    # desired - target value we want to achieve after executing new trade
    # post_trade - value we actually achieved after trade

    # -----
    # TODO:
    # - account.mark_to_market(bid, ask) ?
    # - mark_to_market(account, bid, ask) -> account
    # - models: Account, Position, PositionPnL, Order
    # - managers:
    # -- AccountManager.mark_to_market(account) -> account
    # -- PositionManager.mark_to_market(position, bid, ask), PositionManager.update_pnl...
    # -- ExecutionManager.execute(current_position, new_position, algo)
    # - change logs:
    # -- AccountChangeLog, PositionChangeLog,

    # -----
    # Get prior account and position states
    account_prior = prior_contexts[0].account_post_trade
    position_prior = account_prior.position
    position_pnl_prior = account_prior.position_pnl

    # Mark account and position to market before processing new prediction
    if position_prior is not None:
        price_pre_trade = bid if position_prior.size > 0 else ask
        position_pre_trade = position_prior._replace(current_price=price_pre_trade)
    else:
        position_pre_trade = None

    account_pre_trade = account_prior._replace(position=position_pre_trade)
    size_pre_trade = position_pre_trade.size if position_pre_trade is not None else 0

    # Process new prediction, calculate desired position for the current bar
    position_desired = DesiredPosition(
        proba=proba,
        size=self.sizing_fn(proba) * account_pre_trade.balance
    )

    # Calculate diff with pre_trade position, create orders
    size_diff = position_desired.size - size_pre_trade
    order = Order(
        type=OrderType.market,
        size=size_diff,
        price=ask if size_diff > 0 else bid
    ) if size_diff != 0 else None

    # Execute orders
    costs = self.exchange.costs
    # TODO: Subtract fees directly from balance on each trade
    # TODO: Update position to contain correct size and price
    # - subtract fee from position's realized PnL on each trade
    # - pnl_realized = execution_res.size * (position_pre_trade.entry_price? / execution_res.price - 1) ??
    # - what to do when we reverse position? close previous, open new? how to store?
    # - separate calculation parts into services, MarkToMarket, PositionSizer, Execution, AccountUpdater, whatever
    # * size = old + new
    # * price = how to calculate? old normalized by size + new normalized by size. How to normalize?

    if order.size != 0:
        # The actual trade would be adversely affected by slippage = drift + impact
        slipped_price = order.price * (1 + order.direction * abs(costs.slippage))

        execution_res = ExecutionResult(
            order=order,
            size=order.size,
            price=slipped_price,
            fee=abs(order.size) * costs.taker_fee,
            slippage=slipped_price - order.price,
        )

        if account_pre_trade.position is None:
            pnl_realized = 0
            position_post_trade = Position(
                size=execution_res.size,
                entry_price=execution_res.price,
                current_price=execution_res.price,
                pnl_realized=pnl_realized
            )
        else:
            # Update position state
            # Method 2 from here: https://www.deltastock.com/english/education/average-price.asp
            if position_pre_trade.direction == execution_res.direction:
                # Calculate average entry price if we're adding to the position
                size_pre_trade = position_pre_trade.size
                price_pre_trade = position_pre_trade.entry_price
                size_executed = execution_res.size
                price_executed = execution_res.price
                entry_price = (size_pre_trade * price_pre_trade + size_executed * price_executed) / (size_pre_trade + size_executed)

                pnl_realized = 0  # position wasn't reduced, no pnl realized

            else:
                # Keep the same average price if we're reducing position. Realize PnL (see link above)
                entry_price = position_pre_trade.entry_price
                pnl_realized = 0  # TODO: position was reduced, calculate realized pnl (how?)

            position_post_trade = Position(
                size=position_pre_trade.size + execution_res.size,  # adjust by traded size
                entry_price=entry_price,
                current_price=execution_res.price,
                pnl_realized=position_pre_trade.pnl_realized + pnl_realized
            )
            # add realized pnl to account balance

        account_post_trade = Account(
            balance=account_pre_trade.balance + execution_res.fee + pnl_realized,
            position=position_post_trade
        )

    else:
        execution_res = None
        account_post_trade = account_pre_trade

    return BarContext(
        account_pre_trade=account_pre_trade,
        position_desired=position_desired,
        order=order,
        execution_result=execution_res,
        account_post_trade=account_post_trade
    )


def _contexts(self, pred_probas: np.ndarray, bids: np.ndarray, asks: np.ndarray) -> List[BarContext]:
    account_pre_trade = Account(
        balance=self.initial_equity,
        position=None,
    )

    initial_context = BarContext(
        account_pre_trade=account_pre_trade,
        position_desired=None,
        order=None,
        execution_result=None,
        account_post_trade=account_pre_trade
    )

    contexts = [initial_context]

    for i in range(pred_probas.size):
        context = self._on_proba(pred_probas[i], bids[i], asks[i], prior_contexts=list(reversed(contexts)))
        contexts.append(context)

    return contexts
