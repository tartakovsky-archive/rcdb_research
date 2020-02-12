import numpy as np

from typing import Union, Optional

from ..entities import Probabilities, Predictions


class PredictionSimulator:
    """
    Static class for converting predicted probabilities into class labels
    """

    @staticmethod
    def preds(probas: Probabilities, threshold: float = 0.5,
              direction: str = 'pos', labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}) -> Predictions:
        """
        # TODO: PyCharm autogenerate ours or sklearns docstring format

        :param probas: instance of Probabilities class
        :param threshold: level of probability which needs to be exceeded to signal a class label
        :param direction: one of ['pos', 'neg']
        :param labels: mapping from ['pos', 'neu', 'neg'] directions to class labels
        :return:
        """

        supported_directions = ['pos', 'neg']
        if direction not in supported_directions:
            raise ValueError(
                f'{direction} direction is not supported. Should be one of the following: {supported_directions}'
            )

        p = probas.y_pred_proba if direction == 'pos' else (1 - probas.y_pred_proba)

        y_pred = np.where(p > threshold, labels[direction], labels['neu'])
        preds = Predictions(probas.y_true, y_pred, probas.index)
        return preds.init_metrics(direction, labels)
