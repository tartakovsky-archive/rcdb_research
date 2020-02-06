import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..metrics import PredictionMetrics


class Predictions:
    """
    Class for storing predicted labels of ML models
    """
    ############
    # Initialization
    ############

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, index: np.ndarray = None):

        if index is not None:
            index = index.copy()

            if y_true.size > index.size:
                raise ValueError(f'index.size={index.size} should be >= y_true.size={y_true.size}')

            if y_true.size != index.size:
                logging.warning(
                    f' Last {y_true.size} out of {index.size} '
                    'elements will be taken from index to match y_true size'
                )

            index = index[-y_true.size:]

        if y_pred.size != y_true.size:
            raise ValueError(f"Size of y_pred should be same size with y_true")

        self.y_true = y_true
        self.y_pred = y_pred
        self.index = index

    ############
    # Public methods
    ############
    def init_metrics(self, target_label: Union[str, int] = 'all', neu_label: int = 0) -> 'Predictions':
        self.metric_params = dict(target_label=target_label, neu_label=neu_label)
        self.metrics = PredictionMetrics(self, **self.metric_params)
        return self

    def head(self, n: int) -> 'Predictions':
        """
        Returns copy of Predictions with first n items
        :param n: number of first items
        :return:
        """
        new_preds = Predictions(
            y_true=self.y_true[:n],
            y_pred=self.y_pred[:n],
            index=self.index[:n]
        )

        if hasattr(self, 'metric_params'):
            new_preds.init_metrics(self, **self.metric_params)

        return new_preds

    def tail(self, n: int) -> 'Predictions':
        """
        Returns copy of Predictions with last n items
        :param n: number of last items
        :return:
        """
        new_preds = Predictions(
            y_true=self.y_true[-n:],
            y_pred=self.y_pred[-n:],
            index=self.index[-n:]
        )

        if hasattr(self, 'metric_params'):
            new_preds.init_metrics(self, **self.metric_params)

        return new_preds

    def in_date_range(self, date_start: Optional[str] = None, date_end: Optional[str] = None) -> 'Predictions':
        """
        Returns copy of Predictions with items with indexes between date_start and date_end
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

        sub_y_true = self.y_true[np.isin(self.index, sub_index)]
        sub_y_pred = self.y_pred[np.isin(self.index, sub_index)]

        new_preds = Predictions(
            y_true=sub_y_true,
            y_pred=sub_y_pred,
            index=sub_index,
        )

        if hasattr(self, 'metric_params'):
            new_preds.init_metrics(self, **self.metric_params)

        return new_preds
