# flake8: noqa


import numbers
import itertools
from copy import deepcopy
from warnings import warn
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import logging

from rcdb_research.sampling import sequential_bootstrap
from rcdb_research.feature_importance.utils import cluster_labels_to_clusters

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from sklearn.ensemble.bagging import BaseBagging, BaggingClassifier, BaggingRegressor
    from sklearn.base import ClassifierMixin, RegressorMixin
    from sklearn.ensemble.base import _partition_estimators
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import indices_to_mask
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils import check_random_state, check_array, check_consistent_length, check_X_y
from sklearn.utils._joblib import Parallel, delayed

# from mlfinlab.sampling.bootstrapping import seq_bootstrap, get_ind_matrix

from sklearn.base import clone
from sklearn.ensemble._base import _set_random_states

MAX_INT = np.iinfo(np.int32).max


def _generate_random_features(random_state, bootstrap, n_population, n_samples):
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples,
                                             random_state=random_state)

    return indices


def _generate_bagging_indices(random_state, bootstrap_features, n_features, max_features, n_samples, max_samples, t1,
                              bars_idx, clusters_int):
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    if clusters_int is None:
        feature_indices = _generate_random_features(random_state, bootstrap_features,
                                                    n_features, max_features)
    else:
        if bootstrap_features is True:
            raise ValueError('case not supported: bootstrap_features=True and clusters_int is not None')
        cluster_indices = random_state.choice(len(clusters_int), size=max_features)
        clusters_subsample = [clusters_int[i] for i in cluster_indices]
        feature_indices = [random_state.choice(columns) for columns in clusters_subsample]

    #     sample_indices = seq_bootstrap(ind_mat, sample_length=max_samples, random_state=random_state)  # <- --- ---
    if t1 is not None:
        sample_indices = sequential_bootstrap(t1, bars_idx, max_samples, seed=int(random_state.random() * 1e6))
    else:
        sample_indices = random_state.choice(n_samples, max_samples, replace=True)

    return feature_indices, sample_indices


def _parallel_build_estimators(n_estimators, max_features, max_samples, bootstrap_features, support_sample_weight,
                               base_estimator, estimator_params, X, y, t1, bars_idx, clusters_int, sample_weight,
                               seeds, total_n_estimators, verbose):
    # Retrieve settings
    n_samples, n_features = X.shape

    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []
    estimators_indices = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])
        #         estimator = ensemble._make_estimator(append=False,
        #                                              random_state=random_state)
        estimator = clone(base_estimator)
        estimator.set_params(**estimator_params)
        _set_random_states(estimator, random_state)

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      n_features,
                                                      max_features,
                                                      n_samples,
                                                      max_samples,
                                                      t1,
                                                      bars_idx,
                                                      clusters_int)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            sample_counts = np.bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)
        estimators_indices.append(indices)

    return estimators, estimators_features, estimators_indices


class CSBBBase(BaseBagging, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self,
                 #                  bars_idx,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            bootstrap=True,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        # pylint: disable=invalid-name
        #         self.samples_info_sets = samples_info_sets
        #         self.price_bars = price_bars
        #         self.bars_idx = bars_idx
        #         self.ind_mat = get_ind_matrix(samples_info_sets, price_bars)  # <- --- --- --- -|
        # Used for create get ind_matrix subsample during cross-validation
        #         self.timestamp_int_index_mapping = pd.Series(index=samples_info_sets.index,
        #                                                      data=range(self.ind_mat.shape[1]))

        # self.X_time_index = None  # Timestamp index of X_train

    def fit(self, X, y, sample_weight=None, t1=None, bars_idx=None, clusters=None, labels=None, clusterer=None):
        if (t1 is not None) ^ (bars_idx is not None):
            raise ValueError('both t1 and bars_idx must be specified')
        if clusters is not None:
            flattened_clusters = list(sorted([y for x in clusters for y in x['columns']]))
            if isinstance(X, pd.DataFrame):
                columns = list(sorted(X.columns))
                if flattened_clusters != columns:
                    raise ValueError('clusters content doesn\'t match X.columns')
            elif isinstance(X, np.ndarray):
                if labels is None:
                    raise ValueError('X is np.ndarray and labels is None')
                if flattened_clusters != list(sorted(labels)):
                    raise ValueError('clusters content doesn\'t match labels')
                if len(labels) != X.shape[1]:
                    raise ValueError('labels content doesn\'t match X.shape[1]')
        return self._fit(
            X, y,
            self.max_samples, sample_weight=sample_weight, t1=t1, bars_idx=bars_idx, clusters=clusters, labels=labels,
            clusterer=clusterer
        )

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None, t1=None, bars_idx=None, clusters=None,
             labels=None, clusterer=None):
        #         spans = encode(X[['t0', 't1']].values, self.bars_idx)
        #         X = X.drop(['t0', 't1'], axis=1)
        column_names = deepcopy(X.columns) if isinstance(X, pd.DataFrame) else labels

        self.t1 = t1
        self.bars_idx = bars_idx

        random_state = check_random_state(self.random_state)
        # self.X_time_index = X.index  # Remember X index for future sampling

        # Generate subsample ind_matrix (we need this during subsampling cross_validation)
        #         subsampled_ind_mat = self.ind_mat[:, self.timestamp_int_index_mapping.loc[self.X_time_index]]

        # Convert data (X is required to be 2d and indexable)
        X, y = check_X_y(
            X, y, ['csr', 'csc'], dtype=None, force_all_finite=False,
            multi_output=True
        )
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            check_consistent_length(y, sample_weight)

        if clusterer is not None:
            if clusters is not None:
                logging.warning(f'`clusterer` param is set, ignoring `clusters` param')
            X_ = pd.DataFrame(X, columns=labels)
            clusterer.fit(X_.T)
            clusters = cluster_labels_to_clusters(clusterer.labels_, X_.columns)
        else:
            X_ = pd.DataFrame(X, columns=labels)
            clusters = clusters or [
                dict(name=col, columns=[col])
                for col in X_.columns
            ]

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        # Validate max_samples
        if not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        elif isinstance(self.max_features, np.float):
            if clusters is not None:
                max_features = self.max_features * len(clusters)
            else:
                max_features = self.max_features * self.n_features_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []
            self.sequentially_bootstrapped_samples_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators, self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        clusters_int = None
        if clusters is not None:
            clusters_int = []
            for cluster in clusters:
                clusters_int.append(np.where(np.isin(column_names, cluster['columns']))[0])
        self.clusters_int = clusters_int

        # pylint: disable=C0330
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],  # n_estimators
                self._max_features,  # max_features
                self._max_samples,  # max_samples
                self.bootstrap_features,  # bootstrap_features
                has_fit_parameter(self.base_estimator_, 'sample_weight'),  # support_sample_weight
                self.base_estimator_,  # base_estimator
                {k: getattr(self, k) for k in self.estimator_params},  # estimator_params
                X,  # X
                y,  # y
                t1,  # t1
                bars_idx,  # bars_idx
                clusters_int,  # clusters_int
                sample_weight,  # sample_weight
                seeds[starts[i]:starts[i + 1]],  # seeds
                total_n_estimators,  # total_n_estimators
                verbose=self.verbose  # verbose
            ) for i in range(n_jobs)
        )

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))
        # self.sequentially_bootstrapped_samples_ += list(itertools.chain.from_iterable(
        #     t[2] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self


    def _get_estimators_indices(self):
        # Get drawn indices along both sample and feature axes
        # Operations accessing random_state must be performed identically
        # to those in `_parallel_build_estimators()`
        return Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_generate_bagging_indices)(
                seed,
                self.bootstrap_features,
                self.n_features_,
                self._max_features,
                self._n_samples,
                self._max_samples,
                self.t1,
                self.bars_idx,
                self.clusters_int
            ) for seed in self._seeds
        )
        # for seed in self._seeds:
        #     # Operations accessing random_state must be performed identically
        #     # to those in `_parallel_build_estimators()`
        #     feature_indices, sample_indices = _generate_bagging_indices(
        #         seed,
        #         self.bootstrap_features,
        #         self.n_features_,
        #         self._max_features,
        #         self._n_samples,
        #         self._max_samples,
        #         self.t1,
        #         self.bars_idx,
        #         self.clusters_int
        #     )
        #
        #     yield feature_indices, sample_indices

    #
    # @property
    # def estimators_samples_(self):
    #     # return self.sequentially_bootstrapped_samples_


class CSBBClassifier(CSBBBase, BaggingClassifier, ClassifierMixin):
    """
    Clustered, Sequentially Bootstrapped, Bagging Classifier
    References:
        https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/ensemble/_bagging.py#L432
        https://github.com/hudson-and-thames/mlfinlab/blob/95f7ad59a0e27e99cccd8cec2c6d0e06223fd5cd/mlfinlab/ensemble/sb_bagging.py
    """
    def __init__(self,
                 #                  bars_idx,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            #             bars_idx=bars_idx,
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BaggingClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier())

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_

        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(self.estimators_,
                                                self.sequentially_bootstrapped_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~indices_to_mask(samples, n_samples)

            if hasattr(estimator, "predict_proba"):
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features])

            else:
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        if (predictions.sum(axis=1) == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")

        oob_decision_function = (predictions /
                                 predictions.sum(axis=1)[:, np.newaxis])
        oob_score = accuracy_score(y, np.argmax(predictions, axis=1))

        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score
