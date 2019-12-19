import logging

import numpy as np

from typing import Optional

from .predictions import Predictions


class Probabilities:
    """
    Class for storing predicted probabilities of ML models
    """
    ############
    # Initialization
    ############

    def __init__(self, y_true: np.array, y_pred_proba: np.array, index: np.array = None,
                 label: int = 1, neg_label: int = 0):

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
        self.label = label
        self.neg_label = neg_label

    def predictions(self, threshold: float = 0.5) -> 'Predictions':
        y_pred = np.where(self.y_pred_proba > threshold, self.label, self.neg_label)
        return Predictions(self.y_true, y_pred, self.index)

    def inverted_predictions(self, threshold: float = 0.5) -> 'Predictions':
        y_true = np.where(self.y_true == self.label, self.neg_label, self.label)
        y_pred_proba = 1 - self.y_pred_proba
        y_pred = np.where(y_pred_proba > threshold, self.label, self.neg_label)
        return Predictions(y_true, y_pred, self.index)

    def combined_predictions(self, threshold: float = 0.5, inv_threshold: Optional[float] = None) -> 'Predictions':
        if inv_threshold is None:
            inv_threshold = threshold

        y_true = self.y_true
        inv_y_true = np.where(self.y_true == self.label, self.neg_label, self.label)
        y_pred = np.where(self.y_pred_proba > threshold, self.label, self.neg_label)
        inv_y_pred = np.where((1 - self.y_pred_proba) > inv_threshold, self.label, self.neg_label)

        combined_y_true = y_true - inv_y_true
        combined_y_pred = y_pred - inv_y_pred

        return Predictions(combined_y_true, combined_y_pred, self.index)
