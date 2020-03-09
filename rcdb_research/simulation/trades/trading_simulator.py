from __future__ import annotations

from typing import List, NamedTuple, Optional

import numpy as np
# import pandas as pd
from .account import Account, AccountManager
from .exchange import Exchange, BidAsk
from .execution import ExecutionResult, MMMEA, ExecutionManager, ExecutionAlgo
from .portfolio import Portfolio
from .position import Position, PositionManager, PositionAction
from .sizing import SizingAlgo


class Context(NamedTuple):
    account_pre_trade: Account
    position_desired: Optional[Position]
    actions: List[PositionAction]
    changes: List[ExecutionResult]
    account_post_trade: Account


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
            account=prior_contexts[0].account,
            price=price
        )
        entry_pre_trade = account_pre_trade.portfolio.entry
        position_pre_trade = entry_pre_trade.position if entry_pre_trade is not None else None

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
        account_post_trade = AccountManager.merge_changes(
            account=account_pre_trade,
            changes=executed_changes
        )

        return Context(
            account_pre_trade=account_pre_trade,
            position_desired=desired_position,
            actions=desired_actions,
            changes=executed_changes,
            account_post_trade=account_post_trade
        )

    def _contexts(self, pred_probas: np.ndarray, bidasks: List[BidAsk]) -> List[Context]:
        account_pre_trade = Account(
            exchange=self.exchange,
            balance=self.initial_equity,
            portfolio=Portfolio(entry=None)
        )

        initial_context = Context(
            account_pre_trade=account_pre_trade,
            position_desired=None,
            actions=[],
            changes=[],
            account_post_trade=account_pre_trade
        )

        contexts = [initial_context]

        for i in range(pred_probas.size):
            context = self.on_bar(proba=pred_probas[i], price=bidasks[i], prior_contexts=list(reversed(contexts)))
            contexts.append(context)

        return contexts

    def trades(self):
        pass

#
# def _on_proba(self, proba: float, bid: float, ask: float, prior_contexts: List[BarContext]) -> BarContext:
#     # Typically used modifiers:
#     # prior - last available value of the previous bar
#     # pre_trade - current value before new trade is executed
#     # desired - target value we want to achieve after executing new trade
#     # post_trade - value we actually achieved after trade
#
#     # -----
#     # TODO:
#     # - account.mark_to_market(bid, ask) ?
#     # - mark_to_market(account, bid, ask) -> account
#     # - models: Account, Position, PositionPnL, Order
#     # - managers:
#     # -- AccountManager.mark_to_market(account) -> account
#     # -- PositionManager.mark_to_market(position, bid, ask), PositionManager.update_pnl...
#     # -- ExecutionManager.execute(current_position, new_position, algo)
#     # - change logs:
#     # -- AccountChangeLog, PositionChangeLog,
#
#     # -----
#     # Get prior account and position states
#     account_prior = prior_contexts[0].account_post_trade
#     position_prior = account_prior.position
#     position_pnl_prior = account_prior.position_pnl
#
#     # Mark account and position to market before processing new prediction
#     if position_prior is not None:
#         price_pre_trade = bid if position_prior.size > 0 else ask
#         position_pre_trade = position_prior._replace(current_price=price_pre_trade)
#     else:
#         position_pre_trade = None
#
#     account_pre_trade = account_prior._replace(position=position_pre_trade)
#     size_pre_trade = position_pre_trade.size if position_pre_trade is not None else 0
#
#     # Process new prediction, calculate desired position for the current bar
#     position_desired = DesiredPosition(
#         proba=proba,
#         size=self.sizing_fn(proba) * account_pre_trade.balance
#     )
#
#     # Calculate diff with pre_trade position, create orders
#     size_diff = position_desired.size - size_pre_trade
#     order = Order(
#         type=OrderType.market,
#         size=size_diff,
#         price=ask if size_diff > 0 else bid
#     ) if size_diff != 0 else None
#
#     # Execute orders
#     costs = self.exchange.costs
#     # TODO: Subtract fees directly from balance on each trade
#     # TODO: Update position to contain correct size and price
#     # - subtract fee from position's realized PnL on each trade
#     # - pnl_realized = execution_res.size * (position_pre_trade.entry_price? / execution_res.price - 1) ??
#     # - what to do when we reverse position? close previous, open new? how to store?
#     # - separate calculation parts into services, MarkToMarket, PositionSizer, Execution, AccountUpdater, whatever
#     # * size = old + new
#     # * price = how to calculate? old normalized by size + new normalized by size. How to normalize?
#
#     if order.size != 0:
#         # The actual trade would be adversely affected by slippage = drift + impact
#         slipped_price = order.price * (1 + order.direction * abs(costs.slippage))
#
#         execution_res = ExecutionResult(
#             order=order,
#             size=order.size,
#             price=slipped_price,
#             fee=abs(order.size) * costs.taker_fee,
#             slippage=slipped_price - order.price,
#         )
#
#         if account_pre_trade.position is None:
#             pnl_realized = 0
#             position_post_trade = Position(
#                 size=execution_res.size,
#                 entry_price=execution_res.price,
#                 current_price=execution_res.price,
#                 pnl_realized=pnl_realized
#             )
#         else:
#             # Update position state
#             # Method 2 from here: https://www.deltastock.com/english/education/average-price.asp
#             if position_pre_trade.direction == execution_res.direction:
#                 # Calculate average entry price if we're adding to the position
#                 size_pre_trade = position_pre_trade.size
#                 price_pre_trade = position_pre_trade.entry_price
#                 size_executed = execution_res.size
#                 price_executed = execution_res.price
#                 entry_price = (size_pre_trade * price_pre_trade + size_executed * price_executed) / (size_pre_trade + size_executed)
#
#                 pnl_realized = 0  # position wasn't reduced, no pnl realized
#
#             else:
#                 # Keep the same average price if we're reducing position. Realize PnL (see link above)
#                 entry_price = position_pre_trade.entry_price
#                 pnl_realized = 0  # TODO: position was reduced, calculate realized pnl (how?)
#
#             position_post_trade = Position(
#                 size=position_pre_trade.size + execution_res.size,  # adjust by traded size
#                 entry_price=entry_price,
#                 current_price=execution_res.price,
#                 pnl_realized=position_pre_trade.pnl_realized + pnl_realized
#             )
#             # add realized pnl to account balance
#
#         account_post_trade = Account(
#             balance=account_pre_trade.balance + execution_res.fee + pnl_realized,
#             position=position_post_trade
#         )
#
#     else:
#         execution_res = None
#         account_post_trade = account_pre_trade
#
#     return BarContext(
#         account_pre_trade=account_pre_trade,
#         position_desired=position_desired,
#         order=order,
#         execution_result=execution_res,
#         account_post_trade=account_post_trade
#     )
