import numpy as np
import pandas as pd
import numba


class Trades:
    """
    Class for analyzing trades of quantitative strategies
    """
    ############
    # Initialization
    ############
    """
    Params
    :param directions: pd.Series, direction of the trade, one of [1, 0, -1]
    :param changes: pd.Series, price change inside the bar, close / open -1
    :param fees: pd.Series, fee for each trade, positive for rebates, negative for fees
    """

    def __init__(self,
                 directions: pd.Series,
                 changes: pd.Series,
                 fees: pd.Series):

        if not directions.index.size == changes.index.size:
            raise ValueError('directions.index.size is not equal to changes.index.size')
        if not directions.index.equals(changes.index):
            raise ValueError('directions.index is not equal to changes.index')
        if not directions.index.size == fees.index.size:
            raise ValueError('directions.index.size is not equal to fees.index.size')
        if not directions.index.equals(fees.index):
            raise ValueError('directions.index is not equal to fees.index')

        self.directions = directions
        self.changes = changes
        self.fees = fees

    ############
    # Public interface
    ############
    def returns(self, after_fees=True, sparse=True) -> pd.Series:
        r = self.changes + (self.fees if after_fees else 0.0)
        return r if sparse else r[r != 0]

    def profits(self, after_fees=True, sparse=True) -> pd.Series:
        r = self.returns(after_fees, sparse=sparse)
        return r.where(r > 0, 0.0) if sparse else r[r > 0]

    def losses(self, after_fees=True, sparse=True) -> pd.Series:
        r = self.returns(after_fees, sparse=sparse)
        return r.where(r < 0, 0.0) if sparse else r[r < 0]

    def mean_profit(self, after_fees=True) -> pd.Series:
        return self.profits(after_fees, sparse=False).mean()

    def mean_loss(self, after_fees=True) -> pd.Series:
        return self.losses(after_fees, sparse=False).mean()

    def n_wins(self, after_fees=True) -> int:
        return np.count_nonzero(self.profits(after_fees))

    def n_losses(self, after_fees=True) -> int:
        return np.count_nonzero(self.losses(after_fees))

    def n_trades(self) -> int:
        return np.count_nonzero(self.directions)

    def n_bars(self) -> int:
        return self.directions.size

    def activity(self) -> float:
        return self.n_trades() / self.n_bars()

    def expectancy(self, after_fees=True, activity_weighed=False) -> float:
        win_p = self.n_wins(after_fees) / self.n_trades()
        loss_p = self.n_losses(after_fees) / self.n_trades()
        mean_profit = self.mean_profit(after_fees)
        mean_loss = self.mean_loss(after_fees)

        expectancy = win_p * mean_profit + loss_p * mean_loss

        return expectancy*self.activity() if activity_weighed else expectancy

    def expected_equity(self, initial: int = 100, position_size: float = 0.5,
                        compounded: bool = False, after_fees=True, sparse=True) -> pd.Series:

        index = self.directions.index if sparse else self.directions[self.directions != 0].index
        expectancy = self.expectancy(after_fees, activity_weighed=sparse)
        returns = np.full(index.size, expectancy)

        equity = Trades._calculate_equity(returns, initial, position_size, compounded)

        return pd.Series(equity, index)

    def expected_cum_return(self, initial: int = 100, position_size: float = 0.5,
                            compounded: bool = False, after_fees=True, sparse=True) -> pd.Series:

        return (self.expected_equity(initial, position_size, compounded, after_fees, sparse) / initial) - 1

    def equity(self, initial: int = 100, position_size: float = 0.5,
               compounded: bool = False, after_fees=True, sparse=True) -> pd.Series:

        returns = self.returns(after_fees)
        if not sparse:
            returns = returns[returns != 0]

        equity = Trades._calculate_equity(returns.values, initial, position_size, compounded)
        return pd.Series(equity, returns.index)

    def cum_return(self, initial: int = 100, position_size: float = 0.5,
                   compounded: bool = False, after_fees=True, sparse=True) -> pd.Series:

        equity = self.equity(initial, position_size, compounded, after_fees, sparse)

        return (equity / initial) - 1

    def drawdown(self, initial: int = 100, position_size: float = 0.5,
                 compounded: bool = False, after_fees=True, sparse=True) -> pd.Series:
        equity = self.equity(initial, position_size, compounded, after_fees, sparse)

        return equity / np.maximum.accumulate(equity) - 1

    ############
    # Private functions
    ############
    @staticmethod
    @numba.njit
    def _calculate_equity(returns, initial_equity, position_size, compounded):

        equity = np.empty(returns.size)
        equity[0] = initial_equity
        if compounded:
            for i in range(1, equity.size):
                equity[i] = equity[i - 1] * (1 + returns[i] * position_size)
        else:
            for i in range(1, equity.size):
                _position_size = position_size * initial_equity
                if equity[i - 1] >= _position_size:
                    equity[i] = equity[i - 1] + _position_size * returns[i]
                else:
                    # Stop trading if there is not enough money left for a full position
                    equity[i] = equity[i - 1]
        return equity
