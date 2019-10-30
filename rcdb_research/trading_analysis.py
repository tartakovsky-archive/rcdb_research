import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


class TradingAnalysis:
    class plot:
        @staticmethod
        def curve(array, title=None, xlabel='Observations', ylabel='Value', pos_color=(0.455, 0.678, 0.819),
                  neg_color=(0.956, 0.427, 0.262), fill=False, percent=False, figsize=(16, 4), ax=None):
            size = array.size
            x = list(range(1, size + 1))
            y = array * 100 if percent else array

            y_pos = y.copy()
            y_neg = y.copy()

            y_pos[y_pos <= 0] = np.nan
            y_neg[y_neg > 0] = np.nan

            fig, ax1 = plt.subplots(figsize=figsize) if ax is None else (None, ax)

            ax1.set_frame_on(False)
            ax1.grid(color='lightgray', linestyle='-.', linewidth=0.5)
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f' if percent else '%.2f'))

            ax1.set_title(title)
            ax1.set_xlabel(xlabel, fontsize=12, labelpad=15)
            ax1.set_ylabel(ylabel, fontsize=12, labelpad=15)

            ax1.plot(x, y_pos, color=pos_color)
            ax1.plot(x, y_neg, color=neg_color)

            if fill:
                # c = list(np.where(y > 0, 'blue', 'red'))
                ax1.fill_between(x, 0, y, where=(y >= 0), facecolor=pos_color, alpha=0.5)
                ax1.fill_between(x, 0, y, where=(y < 0), facecolor=neg_color, alpha=0.5)

            if ax is None:
                plt.tight_layout()
                plt.show()

    """
    Class for analyzing trades simulated from ml model predictions
    """

    ############
    # Initialization
    ############
    def __init__(self, cvresult, y_pct_change, fee):
        self.cvres = cvresult
        self.y_pct_change = y_pct_change.tail(self.cvres.y_true.size)
        self.fee = fee
        self.cache = dict()

    ############
    # Public interface
    ############
    @property
    def win_ids(self):
        return np.where(self.cvres.tp() == 1)[0]

    @property
    def loss_ids(self):
        return np.where(self.cvres.fp() == 1)[0]

    @property
    def wins(self):
        return self.y_pct_change[self.win_ids] - self.fee * 2

    @property
    def losses(self):
        return self.y_pct_change[self.loss_ids] - self.fee * 2

    @property
    def returns(self):
        returns = np.zeros(self.cvres.y_true.size)
        returns[self.win_ids] = self.wins
        returns[self.loss_ids] = self.losses
        return returns

    @property
    def mean_profit(self):
        return self.wins.mean()

    @property
    def mean_loss(self, fee=0.0):
        return self.losses.mean()

    def total_return(self, compounded=False):
        if compounded:
            return np.prod(self.returns + 1) - 1
        else:
            return self.returns.sum()

    def cum_return(self, compounded=False):
        key = f'cum_return-{compounded}'

        if key not in self.cache:
            size = self.returns.size

            cum_return = np.zeros(size)
            cum_return[0] = 1

            if compounded:
                for i in range(1, size):
                    cum_return[i] = cum_return[i - 1] * (1 + self.returns[i])
            else:
                for i in range(1, size):
                    cum_return[i] = cum_return[i - 1] + self.returns[i]

            self.cache[key] = cum_return - 1

        return self.cache[key]

    def equity(self, initial=1, compounded=False):
        return initial + initial * self.cum_return(compounded)

    def drawdown(self, compounded=False):
        equity = self.equity(initial=1, compounded=compounded)
        return equity / np.maximum.accumulate(equity) - 1

    ############
    # Plots
    ############
    def plot_performance(self, compounded=False):
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(16, 7), gridspec_kw={'height_ratios': [3, 1]})

        self.plot.curve(self.cum_return(compounded),
                        title='Performance over bars',
                        xlabel=None,
                        ylabel='Gain, %',
                        percent=True,
                        fill=True,
                        ax=ax0)

        self.plot.curve(self.drawdown(compounded),
                        title=None,
                        xlabel='Bars',
                        ylabel='Drawdown, %',
                        percent=True,
                        fill=True,
                        ax=ax1)

        plt.show()
