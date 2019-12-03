import numpy as np
import pandas as pd
import numba

from typing import Union


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
                 directions: np.array,
                 changes: np.array,
                 fees: np.array,
                 index: np.array):

        if not directions.size == index.size:
            raise ValueError('directions.size is not equal to index.size')
        if not changes.size == index.size:
            raise ValueError('changes.size is not equal to index.size')
        if not fees.size == index.size:
            raise ValueError('fees.size is not equal to index.size')

        self.directions = directions
        self.changes = changes
        self.fees = fees
        self.index = index

    ############
    # Public interface
    ############
    def returns(self, after_fees=True, dense=False, raw: bool = False) -> Union[pd.Series, tuple]:
        index = self.index
        returns = self.changes

        if after_fees:
            returns = returns + self.fees

        if dense:
            ids = np.where(returns != 0)
            index = index[ids]
            returns = returns[ids]

        return (returns, index) if raw else pd.Series(returns, index=index)

    def profits(self, after_fees=True, dense=False, raw: bool = False) -> Union[pd.Series, tuple]:
        profits, index = self.returns(after_fees=after_fees, dense=dense, raw=True)

        if dense:
            ids = np.where(profits > 0)
            index = index[ids]
            profits = profits[ids]
        else:
            profits[profits <= 0] = 0.0

        return (profits, index) if raw else pd.Series(profits, index=index)

    def losses(self, after_fees=True, dense=False, raw: bool = False) -> Union[pd.Series, tuple]:
        losses, index = self.returns(after_fees=after_fees, dense=dense, raw=True)

        if dense:
            ids = np.where(losses < 0)
            index = index[ids]
            losses = losses[ids]
        else:
            losses[losses >= 0] = 0.0

        return (losses, index) if raw else pd.Series(losses, index=index)

    def mean_profit(self, after_fees=True) -> float:
        profits, _ = self.profits(after_fees=after_fees, dense=True, raw=True)
        return profits.mean()

    def mean_loss(self, after_fees=True) -> float:
        losses, _ = self.losses(after_fees=after_fees, dense=True, raw=True)
        return losses.mean()

    def n_wins(self, after_fees=True) -> int:
        profits, _ = self.profits(after_fees=after_fees, dense=True, raw=True)
        return profits.size

    def n_losses(self, after_fees=True) -> int:
        losses, _ = self.losses(after_fees=after_fees, dense=True, raw=True)
        return losses.size

    def n_trades(self) -> int:
        return np.count_nonzero(self.directions)

    def win_proba(self, after_fees=True) -> float:
        return self.n_wins(after_fees) / self.n_trades()

    def loss_proba(self, after_fees=True) -> float:
        return self.n_losses(after_fees) / self.n_trades()

    def n_bars(self) -> int:
        return self.directions.size

    def activity(self) -> float:
        return self.n_trades() / self.n_bars()

    def expectancy(self, after_fees=True) -> float:
        win_p = self.win_proba(after_fees)
        loss_p = self.loss_proba(after_fees)
        mean_profit = self.mean_profit(after_fees)
        mean_loss = self.mean_loss(after_fees)

        return win_p * mean_profit + loss_p * mean_loss

    def expected_equity(self, initial: int = 100, position_size: float = 0.5, compounded: bool = False,
                        after_fees=True, dense=False, raw: bool = False) -> Union[pd.Series, tuple]:

        index = self.index
        expectancy = self.expectancy(after_fees=after_fees)

        if dense:
            index = index[self.directions != 0]
        else:
            expectancy = expectancy*self.activity()

        returns = np.full(index.size, expectancy)

        equity = Trades._calculate_equity(returns, initial, position_size, compounded)

        return (equity, index) if raw else pd.Series(equity, index=index)

    def expected_cum_return(self, initial: int = 100, position_size: float = 0.5, compounded: bool = False,
                            after_fees=True, dense=False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.expected_equity(
            initial=initial, position_size=position_size,
            compounded=compounded, after_fees=after_fees, dense=dense, raw=True
        )

        cum_return = (equity / initial) - 1

        return (cum_return, index) if raw else pd.Series(cum_return, index=index)

    def equity(self, initial: int = 100, position_size: float = 0.5, compounded: bool = False,
               after_fees=True, dense=False, raw: bool = False) -> Union[pd.Series, tuple]:

        returns, index = self.returns(after_fees=after_fees, dense=dense, raw=True)

        equity = Trades._calculate_equity(returns, initial, position_size, compounded)

        return (equity, index) if raw else pd.Series(equity, index=index)

    def cum_return(self, initial: int = 100, position_size: float = 0.5,
                   compounded: bool = False, after_fees=True, dense=False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.equity(
            initial=initial, position_size=position_size,
            compounded=compounded, after_fees=after_fees, dense=dense, raw=True
        )

        cum_return = (equity / initial) - 1

        return (cum_return, index) if raw else pd.Series(cum_return, index=index)

    def drawdown(self, initial: int = 100, position_size: float = 0.5,
                 compounded: bool = False, after_fees=True, dense=False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.equity(
            initial=initial, position_size=position_size,
            compounded=compounded, after_fees=after_fees, dense=dense, raw=True
        )

        drawdown = equity / np.maximum.accumulate(equity) - 1

        return (drawdown, index) if raw else pd.Series(drawdown, index=index)

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
