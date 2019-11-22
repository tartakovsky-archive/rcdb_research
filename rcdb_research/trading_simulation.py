import pandas as pd
import numpy as np

from .cross_validators import CVResult


class TradingSimulation:
    """
    Class for simulating trades from ml model predictions
    """
    ############
    # Initialization
    ############
    """
    Params
    :param cvres: CVResult
    :param y_df: pd.DataFrame, required columns: ['open', 'high', 'low', 'change']
    :param initial_equity: int, how much money portfolio had at the beginning, absolute value, e.g. 100 ($)
    :param position_size: float, fraction of initial equity. Stays constant if compounded = False
    :param maker_fee: float
    :param taker_fee: float
    :param entry_order: str, one of ['market', 'limit']
    :param no_reentry: bool, if True - strategy is going to stay in position until the direction of prediction changes
    :param compounded: bool
    """

    def __init__(self,
                 cvres: CVResult,
                 y_df: pd.DataFrame,
                 initial_equity: int = 100,
                 position_size: float = 0.5,
                 maker_fee: float = -0.1/100,
                 taker_fee: float = 0.025/100,
                 entry_order: str = 'market',
                 no_reentry: bool = False,
                 compounded: bool = False):

        # Check that indices are equal, else raise exception

        if not cvres.y_true.index.size == y_df.index.size:
            raise ValueError('cvres.y_true.index.size is not equal to y_df.index.size')
        if not cvres.y_true.index.equals(y_df.index):
            raise ValueError('cvres.y_true.index is not equal to y_df.index')

        # Check that position_size is valid
        if not (0 < position_size <= 1):
            raise ValueError('position_size should be 0 < position_size <= 1.'
                             ' It is a fraction of initial_equity to trade')

        # Check that orders are valid
        allowed_orders = ['market', 'limit']
        if entry_order not in allowed_orders:
            raise ValueError(
                f'entry_order={entry_order}: unknown order. Should be one of the following: {allowed_orders}'
            )

        # Check that y_df has required columns
        required_columns = {'open', 'high', 'low', 'change'}
        missing_columns = list((required_columns & set(y_df.columns)) ^ required_columns)
        if len(missing_columns) > 0:
            raise ValueError(
                f'y_df is missing required columns: {missing_columns}'
            )

        y_pred = cvres.y_pred.copy()
        entry_fee = maker_fee if entry_order == 'limit' else taker_fee
        exit_fee = taker_fee

        fees = pd.Series(0.0, y_pred.index)

        if no_reentry:
            # Simulate not exiting the trade when the next is in the same direction
            for i in range(y_pred.size):
                if y_pred.iloc[i] == 1:
                    # Check if the observation is a beginning or an end of sequence of 1s
                    is_entry = True if i == 0 else y_pred.iloc[i] != y_pred.iloc[i-1]
                    is_exit = True if i == y_pred.size-1 else y_pred.iloc[i] != y_pred.iloc[i+1]

                    # Check whether we could've entered the trade
                    can_enter_with_limit = (y_df['low'].iloc[i] < y_df['open'].iloc[i]-0.5)
                    can_enter = True if entry_order == 'market' else can_enter_with_limit

                    if is_entry and not can_enter:
                        y_pred.iloc[i] = 0
                        fees.iloc[i] = 0.0
                        continue

                    if is_entry and can_enter:
                        fees.iloc[i] += entry_fee

                    if is_exit:
                        fees.iloc[i] += exit_fee
        else:
            # Check if we can enter each separate trade with limit order
            if entry_order == 'limit':
                y_pred[~(y_df['low'] < y_df['open']-0.5)] = 0
            fees[y_pred != 0] = entry_fee + exit_fee

        # Assing properties
        self.cvres = CVResult(y_true=cvres.y_true, y_pred=y_pred, index=cvres.y_true.index)
        self.y_df = y_df

        self.initial_equity = initial_equity
        self.position_size = position_size

        self.fees = fees

        self.compounded = compounded
        self.cache = dict()

    ############
    # Public interface
    ############
    @property
    def wins(self) -> pd.Series:
        mask = self.cvres.tp() == 1
        return self.y_df['change'][mask] + self.fees[mask]

    @property
    def losses(self) -> pd.Series:
        mask = self.cvres.fp() == 1
        return self.y_df['change'][mask] + self.fees[mask]

    @property
    def returns(self) -> pd.Series:
        returns = pd.Series(0.0, self.cvres.y_true.index)
        returns[self.cvres.tp() == 1] = self.wins
        returns[self.cvres.fp() == 1] = self.losses
        return returns

    @property
    def mean_profit(self) -> float:
        return self.wins.mean()

    @property
    def mean_loss(self) -> float:
        return self.losses.mean()

    @property
    def expectancy(self) -> float:
        return self.cvres.precision()*self.mean_profit + (1 - self.cvres.precision())*self.mean_loss

    @property
    def expectancy_times_recall(self) -> float:
        return self.expectancy * self.cvres.recall()

    @property
    def expected_cum_return(self) -> pd.Series:
        values = np.array(range(self.cvres.y_true.size)) * self.expectancy_times_recall * self.position_size
        return pd.Series(values, self.cvres.y_true.index)

    @property
    def equity(self) -> pd.Series:
        key = 'equity'

        if key not in self.cache:
            size = self.returns.size

            equity = pd.Series(0.0, self.returns.index)
            equity[0] = self.initial_equity

            if self.compounded:
                for i in range(1, size):
                    equity[i] = equity[i - 1] * (1 + self.returns[i] * self.position_size)
            else:
                for i in range(1, size):
                    position_size = self.position_size * self.initial_equity
                    if equity[i - 1] >= position_size:
                        equity[i] = equity[i - 1] + position_size * self.returns[i]
                    else:
                        # Stop trading if there is not enough money left for a full position
                        equity[i] = equity[i - 1]

            self.cache[key] = equity

        return self.cache[key]

    @property
    def cum_return(self) -> pd.Series:
        return self.equity / self.initial_equity - 1

    @property
    def drawdown(self) -> pd.Series:
        equity = self.equity
        return equity / np.maximum.accumulate(equity) - 1
