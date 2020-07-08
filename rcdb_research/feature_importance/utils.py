import pandas as pd
import numpy as np

from operator import itemgetter
from itertools import groupby
from typing import List

from sklearn.ensemble import BaggingClassifier
from ..models import CSBBClassifier


def cluster_labels_to_clusters(labels: List[int], columns: List[str]) -> List[dict]:
    tuples = list(zip(labels, columns))
    groups = groupby(tuples, key=itemgetter(0))
    column_groups = [[v for k, v in g] for k, g in groups]

    clusters = [
        dict(
            name=f'{columns[0]}+{len(columns) - 1}',
            columns=columns
        )
        for columns in column_groups
    ]

    return clusters


def feature_importances(fitted_estimator):
    if hasattr(fitted_estimator, 'feature_importances_'):
        return fitted_estimator.feature_importances_

    if isinstance(fitted_estimator, (BaggingClassifier, CSBBClassifier)):
        rows = [
            dict(zip(fts, est.feature_importances_)) for fts, est in zip(fitted_estimator.estimators_features_,
                                                                         fitted_estimator.estimators_)
        ]
        result = pd.DataFrame(rows).mean().sort_index()
        result = result.reindex(np.arange(fitted_estimator.n_features_)).values
        return result
    raise ValueError('estimator type not supported: {}'.format(type(fitted_estimator)))
