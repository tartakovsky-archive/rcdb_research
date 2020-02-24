from __future__ import annotations

from typing import Callable, List

import numpy as np
# import pandas as pd

# from rcdb_research.simulation import Probabilities, Trades

from .models import SupportedExchange, Pair, Fee
from .models import EquityState, TradeSignal, OrderType, Order, ExecutionResult, BarContext


class TradingSimulator:

    ############
    # Initialization
    ############
    def __init__(self,
                 exchange: SupportedExchange,
                 pair: Pair,
                 initial_equity: float,
                 sizing_fn: Callable,
                 fee: Fee,
                 slippage: float,
                 impact: float,
                 compounded: bool = False):
        self.exchange = exchange
        self.pair = pair
        self.initial_equity = initial_equity
        self.sizing_fn = sizing_fn
        self.fee = fee
        self.slippage = slippage
        self.impact = impact
        self.compounded = compounded

    def _on_proba(self, proba: float, bid: float, ask: float, prior_contexts: List[BarContext]) -> BarContext:
        prior_context = prior_contexts[0].post_trade_state
        initial_equity = prior_contexts[-1].pre_trade_state.equity

        # is it correct to mark-to-market at bid?
        # or should we estimate base holdings value as (base_holdings * (1 + fees)) * (bid_price * (1 + impact + slippage))

        pre_trade_state = EquityState(
            quote_holdings=prior_context.quote_holdings,
            base_holdings=prior_context.base_holdings,
            base_holdings_quote_value=(prior_context.base_holdings * (1 + self.fee.taker)) * (bid * (1 + self.impact + self.slippage))
        )

        signal = TradeSignal(
            proba=proba,
            desired_price=(bid + ask) / 2,
            desired_exposure=self.sizing_fn(proba)
        )

        exposure_diff = (signal.desired_exposure - pre_trade_state.exposure)
        size = exposure_diff * initial_equity if self.compounded else exposure_diff * pre_trade_state.equity

        order = Order(
            type=OrderType.market,
            exchange=self.exchange,
            pair=self.pair,
            price=signal.desired_price,
            size=size
        )

        if order.size > 0:  # buy base currency
            # We want to do the trade at midmarket price
            # The actual long trade would be done at ask price and adversely affected by slippage and market impact
            actual_price = ask * (1 + np.abs(self.slippage + self.impact))

            # We spend e.g. 1000 USD to get 998 USD worth of BTC after fees
            execution_res = ExecutionResult(
                order=order,
                avg_price=actual_price,
                base_change=order.size * (1 + self.fee.taker) / actual_price,
                quote_change=-order.size,
            )
        elif order.size < 0:  # sell base currency
            # We want to do the trade at midmarket price
            # The actual short trade would be done at bid price and adversely affected by slippage and market impact
            actual_price = bid * (1 - np.abs(self.slippage + self.impact))

            # We sell e.g. 1000 USD worth of BTC to get 998 USD after fees
            execution_res = ExecutionResult(
                order=order,
                avg_price=actual_price,
                base_change=-(order.size / actual_price),
                quote_change=order.size * (1 + self.fee.taker),
            )
        else:  # do nothing
            return BarContext(
                pre_trade_state=pre_trade_state,
                trade_signal=signal,
                order=order,
                execution_result=None,
                post_trade_state=pre_trade_state
            )

        post_trade_state = EquityState(
            quote_holdings=pre_trade_state.quote_holdings + execution_res.quote_change,
            base_holdings=pre_trade_state.base_holdings + execution_res.base_change,
            base_holdings_quote_value=(pre_trade_state.base_holdings * (1 + self.fee.taker)) * (bid * (1 + self.impact + self.slippage))
        )

        return BarContext(
            pre_trade_state=pre_trade_state,
            trade_signal=signal,
            order=order,
            execution_result=execution_res,
            post_trade_state=post_trade_state
        )

    def _contexts(self, pred_probas: np.ndarray, bids: np.ndarray, asks: np.ndarray) -> List[BarContext]:
        pre_trade_state = EquityState(
            quote_holdings=self.initial_equity,
            base_holdings=0,
            base_holdings_quote_value=0
        )

        initial_context = BarContext(
            pre_trade_state=pre_trade_state,
            trade_signal=None,
            order=None,
            execution_result=None,
            post_trade_state=pre_trade_state
        )

        contexts = np.array([initial_context])

        for i in range(pred_probas.size):
            context = self._on_proba(pred_probas[i], bids[i], asks[i], prior_contexts=contexts[:i + 1])
            np.insert(contexts, 0, context)

        return list(contexts)

    # def trades(self, probas: Probabilities, bids: np.ndarray, asks: np.ndarray) -> Trades:
    #     """
    #
    #     """
    #
    #
    #
    #     return trades
