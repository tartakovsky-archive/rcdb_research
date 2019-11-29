import pandas as pd
import logging

from .predictions import Predictions
from .trades import Trades


class MetricsConverter:

    """
    Params
    :param exchange: str, name of the exchange used to model trades, one of ['bitmex', 'bitfinex', 'binance']
    :param entry_order: str, one of ['market', 'limit']
    :param no_reentry: bool, if True - strategy is going to stay in position until the direction of prediction changes
    :param maker_fee: float
    :param taker_fee: float
    :param slippage: float, average slippage the execution engine incurs with typical position size and exchange
    """

    ############
    # Initialization
    ############
    def __init__(self,
                 exchange: str = 'bitfinex',
                 entry_order: str = 'market',
                 no_reentry: bool = False,
                 maker_fee: float = 0.2/100,
                 taker_fee: float = -0.2/100,
                 slippage: float = -0.025/100):

        # Check that entry_order is valid
        supported_exchanges = ['bitmex', 'bitfinex', 'binance']
        if exchange not in supported_exchanges:
            raise ValueError(
                f'entry_order={exchange}: unknown order. Should be one of the following: {supported_exchanges}'
            )

        expected_fees = (self._exchange_fees[exchange]['taker'], self._exchange_fees[exchange]['maker'])
        selected_fees = (taker_fee, maker_fee)
        if expected_fees != selected_fees:
            logging.warning(
                f'\nExpected fees for {exchange} {expected_fees} do not match selected fees {selected_fees}'
                f'\nMake sure that you are modeling the right thing.'
            )

        # Check that entry_order is valid
        allowed_orders = ['market', 'limit']
        if entry_order not in allowed_orders:
            raise ValueError(
                f'entry_order={entry_order}: unknown order. Should be one of the following: {allowed_orders}'
            )

        self.exchange = exchange
        self.entry_order = entry_order
        self.no_reentry = no_reentry
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage

    ############
    # Public interface
    ############
    def convert(self, predicts: 'Predictions', ohlc: pd.DataFrame) -> 'Trades':
        # Warn if the test data appears to be from a different exchange
        if type(predicts.y_true.index) == pd.DatetimeIndex:
            expected_start_date = pd.Timestamp(self._exchange_history_starts[self.exchange])
            actual_start_date = predicts.y_true.index[0]

            if actual_start_date < expected_start_date:
                logging.warning(
                    f'\nPredicts start on {actual_start_date}.'
                    f'\nThat is earlier than historical data starts for {self.exchange}: {expected_start_date}'
                    f'\nMake sure that you are modeling the right thing.'
                )

        # Check that ohlc has required columns, add 'change' column
        required_columns = {'open', 'high', 'low', 'close'}
        missing_columns = list((required_columns & set(ohlc.columns)) ^ required_columns)
        if len(missing_columns) > 0:
            raise ValueError(
                f'ohlc is missing required columns: {missing_columns}'
            )

        ohlc = ohlc[list(required_columns)].copy()
        ohlc = ohlc[ohlc.index.isin(predicts.y_true.index)]
        ohlc['change'] = ohlc['close'] / ohlc['open'] - 1

        # Calculate entry and exit fees
        if self.entry_order == 'limit':
            entry_fee = self.maker_fee
        else:
            entry_fee = self.taker_fee + self.slippage
        exit_fee = self.taker_fee + self.slippage

        # Simulate whether we can enter into a trade with a limit order
        can_enter_limit_long = ohlc['low'] < (ohlc['open'] - 0.5)
        can_enter_limit_short = ohlc['high'] > (ohlc['open'] + 0.5)

        y_pred = predicts.y_pred.copy()
        fees = pd.Series(0.0, y_pred.index)
        is_entry_flags = pd.Series(False, y_pred.index)
        is_exit_flags = pd.Series(False, y_pred.index)

        if self.no_reentry:
            # Simulate not exiting the trade when the next is in the same direction
            for i in range(y_pred.size):
                trade_direction = y_pred.iloc[i]

                if trade_direction != 0:
                    # Check if the observation is a beginning or an end of a sequence
                    is_entry = True if i == 0 else y_pred.iloc[i] != y_pred.iloc[i-1]
                    is_exit = True if i == y_pred.size-1 else y_pred.iloc[i] != y_pred.iloc[i+1]

                    # Check whether we could've entered the trade
                    if trade_direction == 1:
                        can_enter = can_enter_limit_long.iloc[i] if self.entry_order == 'limit' else True
                    elif trade_direction == -1:
                        can_enter = can_enter_limit_short.iloc[i] if self.entry_order == 'limit' else True

                    if is_entry and not can_enter:
                        y_pred.iloc[i] = 0
                        fees.iloc[i] = 0.0
                        continue

                    if is_entry and can_enter:
                        is_entry_flags.iloc[i] = True

                        fees.iloc[i] += entry_fee

                    if is_exit:
                        is_exit_flags.iloc[i] = True
                        fees.iloc[i] += exit_fee
        else:
            # Check if we can enter each separate trade with limit order
            if self.entry_order == 'limit':
                y_pred[(y_pred == 1) & ~can_enter_limit_long] = 0
                y_pred[(y_pred == -1) & ~can_enter_limit_short] = 0
            fees[y_pred != 0] = entry_fee + exit_fee
            is_entry_flags[y_pred != 0] = True
            is_exit_flags[y_pred != 0] = True

        y_true = predicts.y_true
        change = ohlc['change']

        long_wins = (y_pred == 1) & (y_true == 1)
        long_losses = (y_pred == 1) & (y_true != 1)
        short_wins = (y_pred == -1) & (y_true == -1)
        short_losses = (y_pred == -1) & (y_true != -1)

        pnls = pd.Series(0.0, y_pred.index)
        pnls[long_wins] = change[long_wins].abs()
        pnls[long_losses] = -change[long_losses].abs()
        pnls[short_wins] = change[short_wins].abs()
        pnls[short_losses] = -change[short_losses].abs()

        trades = Trades(y_pred, pnls, fees)

        return trades

    ############
    # Private functions
    ############
    @property
    def _exchange_fees(self):
        return {
            'bitmex': {
                'taker': -0.075/100,
                'maker': 0.025/100,
            },
            'bitfinex': {
                'taker': -0.2/100,
                'maker': -0.2/100,
            },
            'binance': {
                'taker': -0.075/100,
                'maker': -0.075/100,
            },
        }

    @property
    def _exchange_history_starts(self):
        return {
            'bitmex': '2017-10-12',
            'bitfinex': '2013-04-01',
            'binance': '2017-10-27',
        }
