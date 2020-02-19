from typing import Callable

import numpy as np
import pandas as pd

from ...numpy_ext import nans_array

from ..entities import Probabilities, Trades
from .sizing import PositionSizing
from .trading_costs import Fee, Slippage, MarketImpact


class TradingSimulator:

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
        # separated into before and after trade because trade affects the price and spends equity to pay fees, therefore altering both size of
        # quote holdings and paper value of base holdings
        equity_before_trade = np.zeros(size + 1)
        equity_before_trade[0] = self.initial_equity
        equity_after_trade = np.zeros(size + 1)
        equity_after_trade[0] = self.initial_equity

        base_holdings = np.zeros(size + 1)  # number of units of base currency currently owned
        quote_holdings = np.zeros(size + 1)  # number of units of quote currency currently owned
        quote_holdings[0] = self.initial_equity

        base_holdings_quote_value_before_trade = np.zeros(size + 1)  # paper value of base holdings at new_bar event before trade
        base_holdings_quote_value_after_trade = np.zeros(size + 1)  # paper value of base holdings after new_bar event after trade

        exposure_before_trade = np.zeros(size + 1)  # fraction of quote value of capital held in base currency
        exposure_after_trade = np.zeros(size + 1)

        # Trade setup
        desired_exposure = np.zeros(size + 1)
        exposure_diff = np.zeros(size + 1)
        trade_size = np.zeros(size + 1)
        trade_direction = np.zeros(size + 1)

        # Execution prices
        desired_price = nans_array(size + 1)
        actual_price = nans_array(size + 1)

        pnl_quote = np.zeros(size + 1)
        pnl_pct = np.zeros(size + 1)

        for i in range(1, probas.y_pred_proba.size + 1):
            # new bar arrived
            fee_pct = self.fee_fn()['taker']
            slippage = self.slippage_fn()
            impact = self.impact_fn()

            # calculate how much do we have
            base_holdings_quote_value_before_trade[i] = base_holdings[i - 1] * ohlc.open[i]
            equity_before_trade[i] = quote_holdings[i - 1] + base_holdings_quote_value_before_trade[i]
            exposure_before_trade[i] = base_holdings_quote_value_before_trade[i] / equity_before_trade[i]

            # track metrics of previous trade
            pnl_quote[i - i] = equity_before_trade[i] - equity_before_trade[i - 1]
            pnl_pct[i - 1] = equity_before_trade[i] / equity_before_trade[i - 1] - 1

            # decide how much to trade
            desired_exposure[i] = self.sizing_fn(y_pred_proba[i])
            exposure_diff[i] = desired_exposure[i] - exposure_before_trade[i]
            # exposure_diff_direction[i] = 1 if exposure_diff[i] > 0 else -1 if exposure_diff[i] < 0 else 0

            if self.compounded:
                trade_size[i] = exposure_diff[i] * equity_before_trade[i]
            else:
                trade_size[i] = exposure_diff[i] * self.initial_equity

            trade_direction[i] = 1 if trade_size[i] > 0 else -1 if trade_size[i] < 0 else 0

            # do the trade
            # TODO: calculate absolute value of paid fees, slippage and impact in quote currency, store in separate arrays
            if trade_direction[i] == 1:
                # We want to do the trade at open price
                desired_price[i] = ohlc.open[i]
                # The actual price we've got is adversely affected by slippage and market impact
                actual_price[i] = desired_price[i] * (1 + np.abs(slippage + impact))

                # We spend e.g. 1000 USD to get 998 USD worth of BTC after fees
                base_holdings[i] = base_holdings[i - 1] + trade_size[i] * (1 - np.abs(fee_pct)) / actual_price[i]
                quote_holdings[i] = quote_holdings[i - 1] - trade_size[i]

            elif trade_direction[i] == -1:
                desired_price[i] = ohlc.open[i]
                actual_price[i] = desired_price[i] * (1 - np.abs(slippage + impact))

                # We sell e.g. 1000 USD worth of BTC to get 998 USD after fees
                base_holdings[i] = base_holdings[i - 1] - trade_size[i] / actual_price[i]
                quote_holdings[i] = quote_holdings[i - 1] + trade_size[i] * (1 - np.abs(fee_pct))

            else:
                # propagate previous values forward and do nothing
                desired_price[i] = np.nan
                actual_price[i] = np.nan
                base_holdings[i] = base_holdings[i - 1]
                quote_holdings[i] = quote_holdings[i - 1]
                pass

            # store trade results
            base_holdings_quote_value_after_trade[i] = base_holdings[i] * actual_price[i]
            equity_after_trade[i] = quote_holdings[i] + base_holdings_quote_value_after_trade[i]
            exposure_after_trade[i] = base_holdings_quote_value_after_trade[i] / equity_after_trade[i]

        trades = Trades(
            equity=equity_before_trade,
            pnls=pnl_pct,
            metadata=dict(
                equity_before_trade=equity_before_trade,
                equity_after_trade=equity_after_trade,
                base_holdings=base_holdings,
                quote_holdings=quote_holdings,
                base_holdings_quote_value_before_trade=base_holdings_quote_value_before_trade,
                base_holdings_quote_value_after_trade=base_holdings_quote_value_after_trade,
                exposure_before_trade=exposure_before_trade,
                exposure_after_trade=exposure_after_trade,
                desired_exposure=desired_exposure,
                exposure_diff=exposure_diff,
                trade_size=trade_size,
                trade_direction=trade_direction,
                desired_price=desired_price,
                actual_price=actual_price,
                pnl_quote=pnl_quote,
                pnl_pct=pnl_pct,
            )
        )

        return trades
















