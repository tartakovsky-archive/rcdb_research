import pandas as pd
import numpy as np

from natsort import natsorted


def cluster_ids_to_clusters(cluster_ids, labels=None):
    if labels is None:
        labels = np.arange(len(cluster_ids))

    # pre-create cluster list
    clusters = [dict(name=i, columns=[]) for i in np.unique(cluster_ids)]

    # put each label into corresponding cluster
    for i, cid in enumerate(cluster_ids):
        clusters[cid]['columns'].append(labels[i])

    # rename clusters
    for c in clusters:
        c['name'] = f"{c['columns'][0]}+{len(c['columns']) - 1}"

    # natsort clusters by name
    clusters = natsorted(clusters, key=lambda item: item['name'])

    return clusters


def feature_importances(fitted_estimator):
    if hasattr(fitted_estimator, 'feature_importances_'):
        return fitted_estimator.feature_importances_

    from sklearn.ensemble import BaggingClassifier
    from ..models import CSBBClassifier
    if isinstance(fitted_estimator, (BaggingClassifier, CSBBClassifier)):
        rows = [
            dict(zip(fts, est.feature_importances_)) for fts, est in zip(fitted_estimator.estimators_features_,
                                                                         fitted_estimator.estimators_)
        ]
        result = pd.DataFrame(rows).mean().sort_index()
        result = result.reindex(np.arange(fitted_estimator.n_features_)).values
        return result
    raise ValueError('estimator type not supported: {}'.format(type(fitted_estimator)))
