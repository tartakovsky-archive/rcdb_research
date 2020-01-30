import numpy as np
import pandas as pd
import numba

from typing import Union, Optional


class Trades:
    """
    Class for analyzing trades of quantitative strategies
    """
    ############
    # Initialization
    ############
    """
    Params
    :param directions: pd.Series, direction of the trade, one of [1, 0, -1]  for long, flat or short trade
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
    def in_date_range(self, date_start: Optional[str] = None, date_end: Optional[str] = None) -> 'Trades':
        """
        Returns copy of Probabilities with items with indexes between date_start and date_end
        :param date_start: Date to drop observations before
        :param date_end: Date to drop observations after
        :return:
        """

        if not isinstance(self.index, pd.DatetimeIndex):
            raise ValueError(f'index should be an instance of pd.DatetimeIndex to use in_date_range method')

        sub_index = self.index
        if date_start is not None:
            sub_index = sub_index[sub_index >= date_start]
        if date_end is not None:
            sub_index = sub_index[sub_index < date_end]

        sub_directions = self.directions[np.isin(self.index, sub_index)]
        sub_changes = self.changes[np.isin(self.index, sub_index)]
        sub_fees = self.fees[np.isin(self.index, sub_index)]

        return Trades(
            directions=sub_directions,
            changes=sub_changes,
            fees=sub_fees,
            index=sub_index,
        )

    def returns(self, after_fees: bool = True, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:
        index = self.index
        returns = self.changes

        if after_fees:
            returns = returns + self.fees

        if dense:
            ids = (returns != 0)
            index = index[ids]
            returns = returns[ids]

        return (returns, index) if raw else pd.Series(returns, index=index)

    def profits(self, after_fees: bool = True, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:
        profits, index = self.returns(after_fees=after_fees, dense=dense, raw=True)

        if dense:
            ids = (profits > 0)
            index = index[ids]
            profits = profits[ids]
        else:
            profits[profits <= 0] = 0.0

        return (profits, index) if raw else pd.Series(profits, index=index)

    def losses(self, after_fees: bool = True, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:
        losses, index = self.returns(after_fees=after_fees, dense=dense, raw=True)

        if dense:
            ids = (losses < 0)
            index = index[ids]
            losses = losses[ids]
        else:
            losses[losses >= 0] = 0.0

        return (losses, index) if raw else pd.Series(losses, index=index)

    ############
    # Trading metrics
    ############
    def mean_profit(self, after_fees: bool = True) -> float:
        profits, _ = self.profits(after_fees=after_fees, dense=True, raw=True)
        return profits.mean()

    def max_profit(self, after_fees: bool = True) -> float:
        profits, _ = self.profits(after_fees=after_fees, dense=True, raw=True)
        return profits.max()

    def mean_loss(self, after_fees: bool = True) -> float:
        losses, _ = self.losses(after_fees=after_fees, dense=True, raw=True)
        return losses.mean()

    def max_loss(self, after_fees: bool = True) -> float:
        losses, _ = self.losses(after_fees=after_fees, dense=True, raw=True)
        return losses.min()

    def n_wins(self, after_fees: bool = True) -> int:
        profits, _ = self.profits(after_fees=after_fees, dense=True, raw=True)
        return profits.size

    def n_losses(self, after_fees: bool = True) -> int:
        losses, _ = self.losses(after_fees=after_fees, dense=True, raw=True)
        return losses.size

    def n_trades(self) -> int:
        return np.count_nonzero(self.directions)

    def pct_wins(self, after_fees: bool = True) -> float:
        return self.n_wins(after_fees) / self.n_trades()

    def pct_losses(self, after_fees: bool = True) -> float:
        return self.n_losses(after_fees) / self.n_trades()

    def n_bars(self) -> int:
        return self.directions.size

    def activity(self) -> float:
        return self.n_trades() / self.n_bars()

    def expectancy(self, after_fees: bool = True) -> float:
        win_p = self.pct_wins(after_fees)
        loss_p = self.pct_losses(after_fees)
        mean_profit = self.mean_profit(after_fees)
        mean_loss = self.mean_loss(after_fees)

        return win_p * mean_profit + loss_p * mean_loss

    def tail_ratio(self, tail: int = 10, after_fees: bool = True) -> float:
        returns, _ = self.returns(after_fees=after_fees, dense=True, raw=True)

        positive_tail = np.percentile(returns, range(100-tail, 100+1)).mean()
        negative_tail = np.abs(np.percentile(returns, range(0, tail+1)).mean())

        return positive_tail / negative_tail

    ############
    # Equity metrics
    ############
    def expected_equity(self, initial: int = 100, position_size: float = 0.5, compounded: bool = False,
                        after_fees: bool = True, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

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
                            after_fees: bool = True, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.expected_equity(
            initial=initial, position_size=position_size,
            compounded=compounded, after_fees=after_fees, dense=dense, raw=True
        )

        cum_return = (equity / initial) - 1

        return (cum_return, index) if raw else pd.Series(cum_return, index=index)

    def equity(self, initial: int = 100, position_size: float = 0.5, compounded: bool = False,
               after_fees: bool = True, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        returns, index = self.returns(after_fees=after_fees, dense=dense, raw=True)

        equity = Trades._calculate_equity(returns, initial, position_size, compounded)

        return (equity, index) if raw else pd.Series(equity, index=index)

    def cum_return(self, initial: int = 100, position_size: float = 0.5, compounded: bool = False,
                   after_fees: bool = True, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.equity(
            initial=initial, position_size=position_size,
            compounded=compounded, after_fees=after_fees, dense=dense, raw=True
        )

        cum_return = (equity / initial) - 1

        return (cum_return, index) if raw else pd.Series(cum_return, index=index)

    def abs_return(self, initial: int = 100, position_size: float = 0.5,
                   compounded: bool = False, after_fees: bool = True) -> float:

        cum_return, _ = self.cum_return(initial=initial, position_size=position_size, compounded=compounded,
                                        after_fees=after_fees, dense=True, raw=True)
        return cum_return[-1]

    def drawdown(self, initial: int = 100, position_size: float = 0.5, compounded: bool = False,
                 after_fees: bool = True, dense: bool = False, raw: bool = False) -> Union[pd.Series, tuple]:

        equity, index = self.equity(
            initial=initial, position_size=position_size,
            compounded=compounded, after_fees=after_fees, dense=dense, raw=True
        )

        drawdown = equity / np.maximum.accumulate(equity) - 1

        return (drawdown, index) if raw else pd.Series(drawdown, index=index)

    def max_drawdown(self, initial: int = 100, position_size: float = 0.5,
                     compounded: bool = False, after_fees: bool = True) -> float:

        return self.drawdown(initial=initial, position_size=position_size,
                             compounded=compounded, after_fees=after_fees).min()

    def annual_return(self, initial: int = 100, position_size: float = 0.5,
                      compounded: bool = False, after_fees: bool = True) -> float:

        cum_return, index = self.cum_return(initial=initial, position_size=position_size,
                                            compounded=compounded, after_fees=after_fees, dense=True, raw=True)

        if type(index) != pd.DatetimeIndex:
            raise ValueError('trades.index should be of type pd.DatetimeIndex to calculate annualized returns')

        abs_return = cum_return[-1]
        years_passed = (index[-1] - index[0]).total_seconds()/60/60/24/365.25

        if compounded:
            return (1+abs_return)**(1/years_passed) - 1
        else:
            return abs_return / years_passed

    def mar(self, initial: int = 100, position_size: float = 0.5,
            compounded: bool = False, after_fees: bool = True) -> float:
        annual_return = self.annual_return(initial=initial, position_size=position_size,
                                           compounded=compounded, after_fees=after_fees)
        max_drawdown = self.max_drawdown(initial=initial, position_size=position_size,
                                         compounded=compounded, after_fees=after_fees)
        return annual_return / np.abs(max_drawdown)

    def metrics(self, initial: int = 100, position_size: float = 0.5, compounded: bool = False,
                after_fees: bool = True, tail: int = 10, dense: bool = False) -> pd.DataFrame:

        abs_return = self.abs_return(initial=initial, position_size=position_size,
                                     compounded=compounded, after_fees=after_fees)
        annual_return = self.annual_return(initial=initial, position_size=position_size,
                                           compounded=compounded, after_fees=after_fees)
        max_drawdown = self.max_drawdown(initial=initial, position_size=position_size,
                                         compounded=compounded, after_fees=after_fees)
        mar = self.mar(initial=initial, position_size=position_size,
                       compounded=compounded, after_fees=after_fees)

        metrics_dict = dict(
            abs_return=abs_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            mar=mar,
            pct_wins=self.pct_wins(after_fees=after_fees),
            pct_losses=self.pct_losses(after_fees=after_fees),
            n_trades=self.n_trades(),
            expectancy=self.expectancy(after_fees=after_fees) * position_size,
            mean_profit=self.mean_profit(after_fees=after_fees),
            max_profit=self.max_profit(after_fees=after_fees),
            mean_loss=self.mean_loss(after_fees=after_fees),
            max_loss=self.max_loss(after_fees=after_fees),
            tail_ratio=self.tail_ratio(tail=10, after_fees=after_fees),
        )
        metrics_df = pd.DataFrame({**metrics_dict}, index=[0])
        return metrics_df

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
