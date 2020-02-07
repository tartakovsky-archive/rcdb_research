import numpy as np
import pandas as pd
import logging

from ..entities import Predictions, Trades


class TradingSimulator:
    """
    Class for converting Predictions into Trades based on exchange and execution parameters.

    Params
    :param exchange: str, name of the exchange used to model trades, one of ['bitmex', 'bitfinex', 'binance']
    :param entry_order: str, one of ['market', 'limit']
    :param no_reentry: bool, if True - strategy is going to stay in position until the direction of prediction changes
    :param maker_fee: float
    :param taker_fee: float
    :param slippage: float, average slippage the execution engine incurs with typical position size and exchange
    :param labels: dict, class labels for positive, neutral and negative classes
    """

    ############
    # Initialization
    ############
    def __init__(self,
                 exchange: str = 'bitfinex',
                 entry_order: str = 'market',
                 no_reentry: bool = False,
                 maker_fee: float = 0.2 / 100,
                 taker_fee: float = -0.2 / 100,
                 slippage: float = -0.025 / 100,
                 labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}):

        # Check that entry_order is valid
        supported_exchanges = ['bitmex', 'bitfinex', 'binance']
        if exchange not in supported_exchanges:
            raise ValueError(
                f'exchange={exchange}: unknown exchange. Should be one of the following: {supported_exchanges}'
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
        self.labels = labels

    ############
    # Public interface
    ############
    def trades(self, predicts: Predictions, ohlc: pd.DataFrame) -> Trades:
        # Warn if the test data appears to be from a different exchange
        if type(predicts.index) == pd.DatetimeIndex:
            expected_start_date = pd.Timestamp(self._exchange_history_starts[self.exchange])
            actual_start_date = predicts.index[0]

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

        index = predicts.index.copy()
        y_true = predicts.y_true.copy()
        y_pred = predicts.y_pred.copy()
        fees = np.zeros(index.size)

        pos, neu, neg = self.labels['pos'], self.labels['neu'], self.labels['neg']

        ohlc = ohlc[ohlc.index.isin(index)]  # TODO: rewrite selection to numpy
        o, h, l, c = ohlc['open'].values, ohlc['high'].values, ohlc['low'].values, ohlc['close'].values
        change = c / o - 1

        # Simulate whether we can enter into a trade with a limit order
        can_enter_limit_long = l < (o - 0.5)
        can_enter_limit_short = h > (o + 0.5)

        # Calculate entry and exit fees
        if self.entry_order == 'limit':
            entry_fee = self.maker_fee + self.slippage
        else:
            entry_fee = self.taker_fee + self.slippage
        exit_fee = self.taker_fee + self.slippage

        if self.no_reentry:
            # Simulate not exiting the trade when the next is in the same direction
            for i in range(y_pred.size):
                if y_pred[i] == neu:
                    continue

                # Check if the observation is a beginning or an end of a sequence
                is_entry = True if i == 0 else y_pred[i] != y_pred[i - 1]
                is_exit = True if i == y_pred.size - 1 else y_pred[i] != y_pred[i + 1]

                # Check whether we could've entered the trade
                if y_pred[i] == pos:
                    can_enter = can_enter_limit_long[i] if self.entry_order == 'limit' else True
                elif y_pred[i] == neg:
                    can_enter = can_enter_limit_short[i] if self.entry_order == 'limit' else True
                else:
                    can_enter = False

                if is_entry and not can_enter:
                    y_pred[i] = neu
                    fees[i] = 0.0
                    continue

                if is_entry and can_enter:
                    fees[i] += entry_fee

                if is_exit:
                    fees[i] += exit_fee
        else:
            # Check if we can enter each separate trade with limit order
            if self.entry_order == 'limit':
                y_pred[(y_pred == pos) & ~can_enter_limit_long] = neu
                y_pred[(y_pred == neg) & ~can_enter_limit_short] = neu
            fees[y_pred != neu] = entry_fee + exit_fee

        long_wins = (y_pred == pos) & (y_true == pos)
        long_losses = (y_pred == pos) & (y_true != pos)
        short_wins = (y_pred == neg) & (y_true == neg)
        short_losses = (y_pred == neg) & (y_true != neg)

        pnls = np.zeros(index.size)
        pnls[long_wins] = np.abs(change[long_wins])
        pnls[long_losses] = -np.abs(change[long_losses])
        pnls[short_wins] = np.abs(change[short_wins])
        pnls[short_losses] = -np.abs(change[short_losses])

        trades = Trades(y_pred, pnls, fees, index)

        return trades

    ############
    # Private
    ############
    @property
    def _exchange_fees(self):
        return {
            'bitmex': {
                'taker': -0.075 / 100,
                'maker': 0.025 / 100,
            },
            'bitfinex': {
                'taker': -0.2 / 100,
                'maker': -0.2 / 100,
            },
            'binance': {
                'taker': -0.075 / 100,
                'maker': -0.075 / 100,
            },
        }

    @property
    def _exchange_history_starts(self):
        return {
            'bitmex': '2017-10-12',
            'bitfinex': '2013-04-01',
            'binance': '2017-10-27',
        }
