import pandas as pd
from pandas.core.common import flatten
from collections import namedtuple
from copy import deepcopy
import numpy as np

from logging import warning
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state
from sklearn.base import (clone, BaseEstimator, ClassifierMixin, MetaEstimatorMixin)
from tqdm.auto import tqdm

from ..sampling.cv.combinatorial import predict_splits, split_indexes_to_bars
from ..scoring.predictions import score_path_2d
from ..metrics.prediction import bounded_log_loss
from .utils import cluster_labels_to_clusters

import inspect


# TODO:
# add support for fit and predict params
# add support for sample weights
# format return in a way so it's integratable into sklearn pipelines, like CalibrationCV

class RFAClassifierCV(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    def __init__(self,
                 estimator,
                 cv,
                 initial_clusters=None,
                 clusters=None,
                 pooling_fn=None,
                 agglomeration=None,
                 max_clusters=5,
                 min_gain=-np.inf,
                 requires_proba=True,
                 score=bounded_log_loss,
                 verbose=True,
                 n_jobs=1):
        self.estimator = estimator
        self.cv = cv

        cluster_params_present = initial_clusters is not None or clusters is not None or pooling_fn is not None
        agglomeration_param_present = agglomeration is not None
        if agglomeration_param_present and cluster_params_present:
            warning("Parameter 'agglomeration' is not None. Parameters 'initial_clusters', 'clusters', "
                    "and 'pooling_fn' will be ignored")

        self.initial_clusters = initial_clusters
        self.clusters = clusters
        self.pooling_fn = pooling_fn
        self.agglomeration = agglomeration
        self.max_clusters = max_clusters
        self.min_gain = min_gain
        self.requires_proba = requires_proba
        self.score = score
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.selected_clusters = []
        self.shouldAgglomerate = (self.clusters is not None and self.pooling_fn is not None) or (self.agglomeration is not None)

    def fit(self, X, y, **fit_params):
        if self.agglomeration is not None:
            self.agglomeration.fit(X)
            self.initial_clusters = []
            self.clusters = cluster_labels_to_clusters(self.agglomeration.labels_, X.columns)
            self.pooling_fn = self.agglomeration.pooling_fn

        base, best, diff, report = rfa(estimator=self.estimator,
                                       X=X,
                                       y=y,
                                       cv=self.cv,
                                       initial_clusters=self.initial_clusters,
                                       clusters=self.clusters,
                                       pooling_fn=self.pooling_fn,
                                       max_clusters=self.max_clusters,
                                       min_gain=self.min_gain,
                                       fit_params=fit_params,
                                       requires_proba=self.requires_proba,
                                       score=self.score,
                                       verbose=self.verbose,
                                       n_jobs=self.n_jobs)

        self.selected_clusters = best['clusters']

        clusters = deepcopy(self.selected_clusters)
        if self.shouldAgglomerate:
            agg_X = pd.DataFrame(index=X.index)
            for i, cluster in enumerate(clusters):
                agg_X[cluster['name']] = self.pooling_fn(X[cluster['columns']].values)
                cluster['columns'] = [cluster['name']]
            selected_features = agg_X
        else:
            selected_features = X[flatten([c['columns'] for c in clusters])]

        estimator_supports_clusters = 'clusters' in inspect.getfullargspec(self.estimator.fit).args

        if estimator_supports_clusters:
            self.estimator.fit(selected_features, y, clusters=clusters, **fit_params)
        else:
            self.estimator.fit(selected_features, y, **fit_params)

    def predict(self, X):
        if self.shouldAgglomerate:
            agg_X = pd.DataFrame(index=X.index)
            for i, cluster in enumerate(self.selected_clusters):
                agg_X[cluster['name']] = self.pooling_fn(X[cluster['columns']].values)
            selected_features = agg_X
        else:
            selected_features = X[flatten([c['columns'] for c in self.selected_clusters])]

        return self.estimator.predict(selected_features)

    def predict_proba(self, X):
        if self.shouldAgglomerate:
            agg_X = pd.DataFrame(index=X.index)
            for i, cluster in enumerate(self.selected_clusters):
                agg_X[cluster['name']] = self.pooling_fn(X[cluster['columns']].values)
            selected_features = agg_X
        else:
            selected_features = X[flatten([c['columns'] for c in self.selected_clusters])]

        return self.estimator.predict_proba(selected_features)


def rfa(estimator, X, y, cv,
        initial_clusters=None,
        clusters=None,
        pooling_fn=None,
        max_clusters=5,
        min_gain=-np.inf,
        fit_params=None,
        requires_proba=True,
        score=bounded_log_loss,
        verbose=True,
        n_jobs=1):
    fit_params = fit_params or {}

    # Flag to decide whether clusters should be agglomerated before scoring
    shouldAgglomerate = clusters is not None and pooling_fn is not None

    # If clusters is set then the whole cluster would be mutated instead of a single feature
    # If clusters is None then each feature is put into separate cluster
    clusters = clusters or [
        dict(name=col, columns=[col])
        for col in X.columns
    ]

    # If both clustered_subset and poolin_fn is set then feature agglomeration would be performed
    # Clusters would be merged into single features usign the pooling_fn
    if shouldAgglomerate:
        agg_X = pd.DataFrame(index=X.index)
        for i, cluster in enumerate(clusters):
            agg_X[cluster['name']] = pooling_fn(X[cluster['columns']].values)
            cluster['columns'] = [cluster['name']]
        X = agg_X

    selected_clusters = initial_clusters or []

    baseline_result = None
    best_result = None
    report = []
    while True:
        print(f'Iteration #{len(report)}')
        iteration = dict(
            selected_clusters=deepcopy(selected_clusters),
            baseline_score=None if len(report) == 0 else report[-1]['best_score'],
            candidate_clusters=deepcopy([c for c in clusters if c not in selected_clusters]),
            candidate_scores=[],
            best_cluster=None,
            best_score=None,
            gain=None,
        )

        if len(iteration['selected_clusters']) == max_clusters:
            if verbose:
                print('Maximum allowed number of selected clusters reached. Store last result as best and finish.')
            report.append(iteration)
            break
        if len(iteration['candidate_clusters']) == 0:
            if verbose:
                print("We're out of candidate features. Store last result as best and finish.")
            report.append(iteration)
            break

        selected_features = X[flatten([c['columns'] for c in iteration['selected_clusters']])]

        if iteration['baseline_score'] is None:
            # For the first iteration calculate baseline score. For the rest last best_score will be reused
            indexes = cv.split(selected_features)
            splits = split_indexes_to_bars(selected_features, y, indexes)
            preds = predict_splits(estimator, splits,
                                   predict_proba=requires_proba,
                                   clusters=iteration['selected_clusters'],
                                   n_jobs=n_jobs)
            scores = np.array(score_path_2d(preds, score)).ravel()
            iteration['baseline_score'] = np.median(scores)
            baseline_result = dict(
                clusters=iteration['selected_clusters'],
                score=np.median(scores)
            )

        if verbose:
            candidate_clusters = tqdm(
                iteration['candidate_clusters'],
                desc='Processing candidates: '
            )
        else:
            candidate_clusters = iteration['candidate_clusters']

        for cluster in candidate_clusters:
            # For each candidate add it to the baseline feature set, train model on CV
            # Store median score between splits as candidate's score
            combined_clusters = iteration['selected_clusters'] + [cluster]
            combined_features = X[flatten([c['columns'] for c in combined_clusters])]

            indexes = cv.split(combined_features)
            splits = split_indexes_to_bars(combined_features, y, indexes)
            preds = predict_splits(estimator, splits,
                                   predict_proba=requires_proba,
                                   clusters=combined_clusters,
                                   n_jobs=n_jobs)
            scores = np.array(score_path_2d(preds, score)).ravel()
            iteration['candidate_scores'].append(np.median(scores))

        best_candidate_id = np.argmax(iteration['candidate_scores'])
        iteration['best_cluster'] = iteration['candidate_clusters'][best_candidate_id]
        iteration['best_score'] = iteration['candidate_scores'][best_candidate_id]
        iteration['gain'] = iteration['best_score'] - iteration['baseline_score']

        if iteration['gain'] < min_gain:
            if verbose:
                print('The last iteration did not find any clusters that improve score more than min_gain. Finish.')
            report.append(iteration)
            break

        selected_clusters = iteration['selected_clusters'] + [iteration['best_cluster']]
        report.append(iteration)

    if len(report) > 0:
        best_result = dict(
            clusters=report[-1]['selected_clusters'],
            score=report[-1]['baseline_score']
        )
    else:
        best_result = baseline_result

    diff = dict(
        clusters=[c for c in best_result['clusters'] if c not in baseline_result['clusters']],
        score=best_result['score'] - baseline_result['score']
    )
    return baseline_result, best_result, diff, report
