from typing import Callable

import numpy as np
import pandas as pd
import logging

from ...numpy_ext import nans_array

from ..entities import Probabilities, Trades
from .sizing import PositionSizing
from .trading_costs import TradingCosts, Fee, Slippage, MarketImpact


class TradingSimulatorNew:

    ############
    # Initialization
    ############
    def __init__(self,
                 initial_equity: float,
                 sizing_fn: Callable = PositionSizing.percent(percent=0.5, threshold=0.6),
                 fee_fn: Callable = Fee.bitfinex(),
                 slippage_fn: Callable = Slippage.percent(-0.1),
                 impact_fn: Callable = MarketImpact.percent(-0.1),
                 compounded: bool = False):
        self.initial_equity = initial_equity
        self.sizing_fn = sizing_fn
        self.fee_fn = fee_fn
        self.slippage_fn = slippage_fn
        self.impact_fn = impact_fn
        self.compounded = compounded

    def trades(self, probas: Probabilities, y_ohlc: pd.DataFrame) -> Trades:
        """
        Models trades on spot exchange (not futures exposure in contracts)

        pos_size_usd = sizing_fn(current_equity)
        entry_size_usd = pos_size_usd * (1 + trading_costs)
        paper_pnl_size_usd = entry_size_usd * (1 + y_change)
        exit_size_usd = paper_pnl_size_usd  * (1 + trading_costs)
        trade_pnl_pct = exit_size_usd / pos_size_usd - 1
        equity_change_pct = (exit_size_usd - pos_size_usd) / current_equity
        """

        y_pred_proba = np.insert(probas.y_pred_proba, 0, 0)
        size = y_pred_proba.size
        ohlc = y_ohlc.tail(size + 1)

        # Capital
        # equity = value of equity in quote currency, quote currency holdings + current paper value of base currency holdings
        # separated into before and after trade because trade affects the price and spends equity to pay fees,
        # therefore altering both size of quote holdings and paper value of base holdings
        equity_before_trade = np.zeros(size + 1)
        equity_before_trade[0] = self.initial_equity
        equity_after_trade = np.zeros(size + 1)
        equity_after_trade[0] = self.initial_equity

        base_holdings = np.zeros(size + 1) # number of units of base currency currently owned
        quote_holdings = np.zeros(size + 1) # number of units of quote currency currently owned
        quote_holdings[0] = self.initial_equity

        base_holdings_quote_value_before_trade = np.zeros(size + 1) # paper value of base holdings at new_bar event before trade
        base_holdings_quote_value_after_trade = np.zeros(size + 1) # paper value of base holdings after new_bar event after trade

        exposure_before_trade = np.zeros(size + 1) # fraction of quote value of capital held in base currency
        exposure_after_trade = np.zeros(size + 1)

        # Trade setup
        desired_exposure = np.zeros(size + 1)
        exposure_diff = np.zeros(size + 1)
        exposure_diff_direction = np.zeros(size + 1)
        trade_size = np.zeros(size + 1)
        trade_direction = np.zeros(size + 1)

        # Execution prices
        desired_entry_price = nans_array(size + 1)
        actual_entry_price = nans_array(size + 1)
        desired_exit_price = nans_array(size + 1)
        virtual_exit_price = nans_array(size + 1)  # we do not actually exit at close but virtually track equity change

        exposure_base = np.zeros(size + 1)
        exposure_direction = np.zeros(size + 1)

        for i in range(1, probas.y_pred_proba.size + 1):
            fee = self.fee_fn()['taker']
            slippage = self.slippage_fn()
            impact = self.impact_fn()

            # new_bar_event, before trade
            # - get current quote value of base holdings
            # - get current exposure pct
            # - get desired exposure pct
            # - get desired - current exposure diff
            # - get exposure diff direction
            base_holdings_quote_value_before_trade[i] = base_holdings[i-1] * ohlc.open[i]
            equity_before_trade[i] = quote_holdings[i-1] + base_holdings_quote_value_before_trade[i]
            exposure_before_trade[i] = base_holdings_quote_value_before_trade[i] / equity_before_trade[i]

            desired_exposure[i] = self.sizing_fn(y_pred_proba[i])
            exposure_diff[i] = desired_exposure[i] - exposure_before_trade[i]
            exposure_diff_direction[i] = 1 if exposure_diff[i] > 0 else -1 if exposure_diff[i] < 0 else 0


            # do the trade
            # - calculate absolute value of paid fees, slippage and impact in quote currency, store in separate arrays
            if exposure_diff_direction[i] == 1:
                # buy base currency
                # - calculate how much of quote currency we should spend before fees and slippage
                #   to buy amount of base currency to bring it's quote value at slipped price to desired exposure level
                # - or just spend exposure_diff * equity of quote currency into the right direction?
                desired_entry_price[i] = ohlc.open[i]
                actual_entry_price[i] = desired_entry_price[i] * (1 + np.abs(slippage + impact))
                base_holdings[i] = base_holdings[i - 1] + bought_base_units
                quote_holdings[i] = quote_holdings[i - 1] - spent_quote_units

            elif exposure_diff_direction[i] == -1:
                # sell (potentially short) base currency
                # - calculate how much of base currency we should sell before fees and slippage
                #   to buy amount of quote currency to bring it's value at slipped price to desired exposure level
                # - or just spend exposure_diff * equity of quote currency into the right direction?
                desired_entry_price[i] = ohlc.open[i]
                actual_entry_price[i] = desired_entry_price[i] * (1 - np.abs(slippage + impact))
                base_holdings[i] = base_holdings[i - 1] - spent_base_units
                quote_holdings[i] = quote_holdings[i - 1] - bought_quote_units

            else:
                # propagate previous values forward and do nothing
                desired_entry_price[i] = np.nan
                base_holdings[i] = base_holdings[i - 1]
                quote_holdings[i] = quote_holdings[i - 1]
                pass

            base_holdings_quote_value_after_trade[i] = base_holdings[i] * actual_entry_price[i]
            equity_after_trade[i] = quote_holdings[i] + base_holdings_quote_value_after_trade[i]
            exposure_after_trade[i] = base_holdings_quote_value_after_trade[i] / equity_after_trade[i]
            #
            # quote_value_of_current_base_exposure = exposure_base[i - 1] * ohlc.open[i]  # store somewhere
            #
            # trade_size[i] = desired_exposure[i] - quote_value_of_current_base_exposure #!!! desired_exposure * equity
            # trade_direction[i] = 1 if trade_size[i] > 0 else -1 if trade_size[i] < 0 else 0
            # desired_entry_price[i] = ohlc.open[i]

            # if trade_direction[i] == 1:
            #     actual_entry_price[i] = desired_entry_price[i] * (1 + np.abs(slippage + impact))
            #     base_holdings[i] = base_holdings[i - 1] + trade_size[i] * (1 - np.abs(fee)) / actual_entry_price[i]
            #     holdings_base_value_in_quote[i] = base_holdings[i] * actual_entry_price[i]
            #     quote_holdings[i] = quote_holdings[i - 1] - trade_size[i]
            #     equity[i] = quote_holdings[i] + holdings_base_value_in_quote[i]
            #
            # elif trade_direction[i] == -1:
            #     actual_entry_price[i] = desired_entry_price[i] * (1 - np.abs(slippage + impact))
            #     base_holdings[i] = base_holdings[i - 1] -
            #     pass
            #
            # 
            #
            # exposure_base[i] = trade_size[i] * (1 - np.abs(fee)) / actual_entry_price[i] + exposure_base[i - 1]
            # exposure_direction[i] = 1 if exposure_base[i] > 0 else -1 if exposure_base[i] < 0 else 0

            desired_exit_price[i] = ohlc.close[i]
            virtual_exit_price[i] = desired_exit_price[i] * (1 - np.abs(slippage + impact)) if exposure_direction[i] == 1 \
                else desired_exit_price[i] * (1 + np.abs(slippage + impact)) if exposure_direction[i] == -1 \
                else np.nan


            # entry_price[i] =
            # TODO: model through open / close prices
            # TODO: separate equity into equity_quote and equity_base_paper_value, plot value of quote and base holdings separately on stacked chart
            # traded_size[i] = size_diff[i] * (1 + self.trading_cost_fn())
            # exposure[i] = traded_size[i] + traded_size[i-1]

        # Calculate position sizes
        # pos_sizes = np.array([self.sizing_fn(self.initial_equity, p) for p in probas.y_pred_proba])
        # pos_sizes = np.where(pos_sizes > 0, pos_sizes, 0)
        # neg_sizes = np.array([self.sizing_fn(self.initial_equity, 1 - p) for p in probas.y_pred_proba])
        # neg_sizes = np.where(neg_sizes > 0, neg_sizes, 0)
        # pos_sizes_usd = pos_sizes - neg_sizes
        #
        # sizediffs = np.diff(np.insert(pos_sizes_usd, 0, 0))
        #
        # trading_costs = sizediffs * np.array([self.trading_costs.get_cost() for _ in probas.y_pred_proba])
        #
        # directions = np.where(sizes > 0, 1, np.where(sizes < 0, -1, 0))
        # changes = y_change * sizes
        #
        # return Trades(directions, changes, trading_costs, probas.index)
        return 0
