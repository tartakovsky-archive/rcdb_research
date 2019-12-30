import numpy as np

from typing import Optional

from ..entities import Probabilities, Predictions


class PredictionSimulator:
    """
    Class for converting probabilities of positive class into class labels

    Params:
    :param labels: dict, class labels for positive, neutral and negative classes
    """
    ############
    # Initialization
    ############

    def __init__(self, labels: dict = {'pos': 1, 'neu': 0, 'neg': -1}):
        self.labels = labels

    def pos_preds(self, probas: 'Probabilities', threshold: float = 0.5) -> 'Predictions':

        y_pred = np.where(probas.y_pred_proba > threshold, self.labels['pos'], self.labels['neu'])
        return Predictions(probas.y_true, y_pred, probas.index)

    def neg_preds(self, probas: 'Probabilities', threshold: float = 0.5) -> 'Predictions':

        inv_y_true = np.where(probas.y_true == self.labels['pos'], self.labels['neu'], self.labels['neg'])
        inv_y_pred_proba = 1 - probas.y_pred_proba
        inv_y_pred = np.where(inv_y_pred_proba > threshold, self.labels['neg'], self.labels['neu'])
        return Predictions(inv_y_true, inv_y_pred, probas.index)

    def combined_preds(self, probas: 'Probabilities', threshold: float = 0.5,
                       inv_threshold: Optional[float] = None) -> 'Predictions':

        if inv_threshold is None:
            inv_threshold = threshold

        y_true = probas.y_true
        inv_y_true = np.where(probas.y_true == self.labels['pos'], self.labels['neu'], self.labels['neg'])
        y_pred = np.where(probas.y_pred_proba > threshold, self.labels['pos'], self.labels['neu'])
        inv_y_pred = np.where((1 - probas.y_pred_proba) > inv_threshold, self.labels['neg'], self.labels['neu'])

        def combine_labels(label, inv_label):
            if label == self.labels['pos'] and inv_label == self.labels['neg']:
                return self.labels['neu']
            elif label == self.labels['neg'] and inv_label == self.labels['pos']:
                return self.labels['neu']
            elif label == self.labels['pos'] or inv_label == self.labels['pos']:
                return self.labels['pos']
            elif label == self.labels['neg'] or inv_label == self.labels['neg']:
                return self.labels['neg']
            else:
                return self.labels['neu']

        combined_y_true = np.apply_along_axis(lambda x: combine_labels(*x), 0, [y_true, inv_y_true])
        combined_y_pred = np.apply_along_axis(lambda x: combine_labels(*x), 0, [y_pred, inv_y_pred])

        return Predictions(np.array(combined_y_true), np.array(combined_y_pred), probas.index)
