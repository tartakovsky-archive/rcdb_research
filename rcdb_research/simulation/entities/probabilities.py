import logging

import numpy as np


class Probabilities:
    """
    Class for storing predicted probabilities of ML models
    """
    ############
    # Initialization
    ############

    def __init__(self, y_true: np.array, y_pred_proba: np.array, index: np.array = None):

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
