from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from typing import Optional


class Probabilities:
    """
    Class for storing predicted probabilities of ML models
    """

    ############
    # Initialization
    ############

    def __init__(self, y_true: np.ndarray, y_pred_proba: np.ndarray, index: np.ndarray = None):

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

        if y_pred_proba.size != y_true.size:
            raise ValueError(f"Size of y_pred_proba should be same size with y_true")

        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.index = index

        self.metrics = None
        self.metric_params = None

    ############
    # Public methods
    ############
    def init_metrics(self, labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}) -> Probabilities:
        from ..metrics import ProbabilityMetrics

        self.metric_params = dict(labels=labels)
        self.metrics = ProbabilityMetrics(self, **self.metric_params)
        return self

    def head(self, n: int) -> Probabilities:
        """
        Returns copy of Probabilities with first n items
        :param n: number of first items
        :return:
        """
        new_probas = Probabilities(
            y_true=self.y_true[:n],
            y_pred_proba=self.y_pred_proba[:n],
            index=self.index[:n]
        )

        if self.metric_params is not None:
            new_probas.init_metrics(**self.metric_params)

        return new_probas

    def tail(self, n: int) -> Probabilities:
        """
        Returns copy of Probabilities with last n items
        :param n: number of last items
        :return:
        """
        new_probas = Probabilities(
            y_true=self.y_true[-n:],
            y_pred_proba=self.y_pred_proba[-n:],
            index=self.index[-n:]
        )

        if self.metric_params is not None:
            new_probas.init_metrics(**self.metric_params)

        return new_probas

    def in_date_range(self, date_start: Optional[str] = None, date_end: Optional[str] = None) -> Probabilities:
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

        sub_y_true = self.y_true[np.isin(self.index, sub_index)]
        sub_y_pred_proba = self.y_pred_proba[np.isin(self.index, sub_index)]

        new_probas = Probabilities(
            y_true=sub_y_true,
            y_pred_proba=sub_y_pred_proba,
            index=sub_index,
        )

        if self.metric_params is not None:
            new_probas.init_metrics(**self.metric_params)

        return new_probas
