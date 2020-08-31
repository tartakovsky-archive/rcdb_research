import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, make_scorer
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from rcdb_research.feature_importance import EFI, MDA, MDI, NMI, cluster_ids_to_clusters
from rcdb_research.feature_importance.utils import feature_importances


def bounded_log_loss(y_true: np.ndarray,
                     y_proba: np.ndarray,
                     bounds=(0.692, 0.684),
                     sample_weight=None) -> float:
    logloss = log_loss(y_true, y_proba, sample_weight=sample_weight)
    return (logloss - bounds[0]) / (bounds[1] - bounds[0])


bounded_log_loss_scorer = make_scorer(bounded_log_loss, needs_proba=True)


def test_usage_example(Xy):
    X, y = Xy
    clusterer = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=0.75)
    clusterer.fit(X.T)
    clusters = cluster_ids_to_clusters(clusterer.labels_, X.columns)
    m1_clf = RandomForestClassifier()

    imp = EFI([
        MDA(m1_clf, bounded_log_loss_scorer, KFold()),
        MDI(m1_clf),
        NMI()
    ])
    _ = imp.fit(X, y, clusters=clusters)

    df = imp.feature_importances_df_

    for c in map(lambda c: c.split('+')[0], df.index.values):
        assert c in X.columns
    assert set(df.columns) == {'mean', 'std', 'rank'}


def test_feature_importances(Xy):
    X, y = Xy
    clf = BaggingClassifier(base_estimator=RandomForestClassifier())
    clf.fit(X, y)

    assert len(feature_importances(clf)) == len(X.columns)
