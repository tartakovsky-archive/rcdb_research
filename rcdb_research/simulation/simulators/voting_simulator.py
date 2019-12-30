import numpy as np

from typing import List, Optional

from ..entities import Probabilities, Predictions


class VotingSimulator:
    """
    Class for simulating voting on a set of predictions from classifiers

    Params:
    :param labels: dict, class labels for positive, neutral and negative classes
    """
    ############
    # Initialization
    ############

    def __init__(self, labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}):
        self.labels = labels

    ############
    # Public interface
    ############
    def plurality_voting(self, preds_arr: List['Predictions'], weights: Optional[List[float]] = None) -> 'Predictions':

        y_preds = np.array([p.y_pred for p in preds_arr])
        y_preds[y_preds == self.labels['pos']] = 1
        y_preds[y_preds == self.labels['neu']] = 0
        y_preds[y_preds == self.labels['neg']] = -1

        # Vote
        if weights is None:
            weights = 1

        weighed_y_preds = np.apply_along_axis(lambda x: x*weights, 0, y_preds)

        y_pred_result = np.sum(weighed_y_preds, axis=0)

        # Label results
        y_pred_result[y_pred_result > 0] = self.labels['pos']
        y_pred_result[y_pred_result == 0] = self.labels['neu']
        y_pred_result[y_pred_result < 0] = self.labels['neg']

        return Predictions(preds_arr[0].y_true, y_pred_result, preds_arr[0].index)

    def no_opposition_voting(self, preds_arr: List['Predictions']) -> 'Predictions':

        y_preds = np.array([p.y_pred for p in preds_arr])

        # Vote
        def bar_vote(bar_preds):
            if np.isin(self.labels['pos'], bar_preds) and not np.isin(self.labels['neg'], bar_preds):
                return self.labels['pos']
            elif np.isin(self.labels['neg'], bar_preds) and not np.isin(self.labels['pos'], bar_preds):
                return self.labels['neg']
            else:
                return self.labels['neu']

        y_pred_result = np.apply_along_axis(bar_vote, axis=0, arr=y_preds)

        return Predictions(preds_arr[0].y_true, y_pred_result, preds_arr[0].index)

    def majority_voting(self, preds_arr: List['Predictions'], weights: Optional[List[float]] = None) -> 'Predictions':

        y_preds = np.array([p.y_pred for p in preds_arr])
        y_preds[y_preds == self.labels['pos']] = 1
        y_preds[y_preds == self.labels['neu']] = 0
        y_preds[y_preds == self.labels['neg']] = -1

        # Vote
        y_pred_result = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=weights)),
            axis=0, arr=y_preds
        )

        # Label results
        y_pred_result[y_pred_result > 0] = self.labels['pos']
        y_pred_result[y_pred_result == 0] = self.labels['neu']
        y_pred_result[y_pred_result < 0] = self.labels['neg']

        return Predictions(preds_arr[0].y_true, y_pred_result, preds_arr[0].index)

    def soft_voting(self, probas_arr: List['Probabilities'], weights: Optional[List[float]] = None) -> 'Probabilities':

        y_pred_probas = np.array([p.y_pred_proba for p in probas_arr])

        avg_y_pred_proba = np.average(y_pred_probas, axis=0, weights=weights)

        return Probabilities(probas_arr[0].y_true, avg_y_pred_proba, probas_arr[0].index)
