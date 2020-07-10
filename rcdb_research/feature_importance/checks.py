import logging

import pandas as pd

from sklearn.utils import check_X_y

from ..feature_importance.utils import cluster_ids_to_clusters


def check_X_y_labels(X, y, labels):
    if labels is None:
        labels = X.columns if hasattr(X, 'columns') else list(range(X.shape[1]))
    index = X.index if hasattr(X, 'index') else list(range(X.shape[0]))
    X, y = check_X_y(X, y)
    X = pd.DataFrame(X, index=index, columns=labels)
    y = pd.Series(y, index=index)
    return X, y, labels, index


def check_clusters(X, clusterer, clusters, labels):
    # If either clusterer or clusters are set, then feature clustering would be performad
    if clusterer is not None:
        if clusters is not None:
            logging.warning(f'`clusterer` param is set, overriding `clusters` param')
        clusterer.fit(X.T)
        clusters = cluster_ids_to_clusters(clusterer.labels_, labels)
    elif clusters is not None:
        clusters = clusters
    else:
        clusters = [dict(name=col, columns=[col]) for col in labels]

    return clusters
