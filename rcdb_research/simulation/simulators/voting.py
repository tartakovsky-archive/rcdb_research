import numpy as np

from typing import List, Optional

from ..entities import Probabilities, Predictions


class Voting:
    """
    Static class for simulating voting on a set of predictions from classifiers
    """

    @staticmethod
    def majority(preds_arr: List[Predictions], weights: Optional[List[float]] = None,
                 labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}) -> Predictions:
        """
        Equivalent of sklearn.VotingClassifier(voting='hard')

        1 if there are more 'for' votes than 'against' and 'neutral' together
        -1 if there are more 'against' votes than 'for' and 'neutral' together
        0 if there is no clear majority

        :param preds_arr: list of Predictions objects
        :param weights: weights to pass to np.argmax
        :param labels:
        :returns: Predictions object with winning predictions
        """

        y_preds = np.array([p.y_pred for p in preds_arr])
        y_preds[y_preds == labels['pos']] = 1
        y_preds[y_preds == labels['neu']] = 0
        y_preds[y_preds == labels['neg']] = -1

        # Vote
        y_pred_result = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=weights)),
            axis=0, arr=y_preds
        )

        # Label results
        y_pred_result[y_pred_result > 0] = labels['pos']
        y_pred_result[y_pred_result == 0] = labels['neu']
        y_pred_result[y_pred_result < 0] = labels['neg']

        return Predictions(preds_arr[0].y_true, y_pred_result, preds_arr[0].index)

    @staticmethod
    def plurality(preds_arr: List[Predictions], weights: Optional[List[float]] = None,
                  labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}) -> Predictions:
        """
        1 if there are more 'for' votes than 'against' and 'neutral' together
        -1 if there are more 'against' votes than 'for' and 'neutral' together
        0 if there is no clear majority

        :param preds_arr: list of Predictions objects
        :param weights: weights to multiply predictions by
        :param labels:
        :returns: Predictions object with winning predictions
        """

        y_preds = np.array([p.y_pred for p in preds_arr])
        y_preds[y_preds == labels['pos']] = 1
        y_preds[y_preds == labels['neu']] = 0
        y_preds[y_preds == labels['neg']] = -1

        # Vote
        if weights is None:
            weights = 1

        weighed_y_preds = np.apply_along_axis(lambda x: x * weights, 0, y_preds)

        y_pred_result = np.sum(weighed_y_preds, axis=0)

        # Label results
        y_pred_result[y_pred_result > 0] = labels['pos']
        y_pred_result[y_pred_result == 0] = labels['neu']
        y_pred_result[y_pred_result < 0] = labels['neg']

        return Predictions(preds_arr[0].y_true, y_pred_result, preds_arr[0].index)

    @staticmethod
    def soft(probas_arr: List[Probabilities], weights: Optional[List[float]] = None,
             labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}) -> Probabilities:
        """
        Equivalent of sklearn.VotingClassifier(voting='soft').predict_proba

        Returns mean of all predicted probabilities for observation. Optionally can be weighed.

        :param probas_arr: list of Probabilities objects
        :param weights: weights to pass to np.average
        :param labels:
        :returns: Probabilities object with mean probas
        """

        y_pred_probas = np.array([p.y_pred_proba for p in probas_arr])

        avg_y_pred_proba = np.average(y_pred_probas, axis=0, weights=weights)

        probas = Probabilities(probas_arr[0].y_true, avg_y_pred_proba, probas_arr[0].index)

        return probas
