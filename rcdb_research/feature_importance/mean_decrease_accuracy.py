import pandas as pd
import numpy as np

from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state
from tqdm.auto import tqdm


def mda(estimator, X, y, cv, clusters=None,
        n_permutations=10, pooling_fn=None,
        fit_params=None, score_params=None,
        scoring=None, random_state=1, raw=False, verbose=True):

    scoring = check_scoring(estimator, scoring)
    rs = check_random_state(random_state)
    fit_params = fit_params or {}
    score_params = score_params or {}

    # Flag to decide whether clusters should be agglomerated before scoring
    shouldAgglomerate = clusters is not None and pooling_fn is not None

    # Handle *_sample_weight in params to support sklearn.Pipelines
    sw_train_name, sw_train = next(
        (kv for kv in fit_params.items() if 'sample_weight' in kv[0]),
        (None, None)
    )
    _ = fit_params.pop(sw_train_name, None)
    sw_score_name, sw_score = next(
        (kv for kv in score_params.items() if 'sample_weight' in kv[0]),
        (None, None)
    )
    _ = score_params.pop(sw_score_name, None)

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

    baseline_scores = []  # [n_folds] of floats
    feature_scores = [[] for _ in clusters]  # [n_folds] of [(n_features * n_permutations)]

    # Split data. Show progress bar if verbose
    splits = cv.split(X=X)
    enumerate_splits = enumerate(tqdm(splits, desc='Splits processed: ')) if verbose else enumerate(splits)

    for i, (train, test) in enumerate_splits:  # for split
        # Train the model on split's train set
        sw_train_dict = {sw_train_name: sw_train[train]} if sw_train_name is not None else {}
        model = estimator.fit(X=X.values[train], y=y.values[train], **sw_train_dict, **fit_params)

        # Get baseline score for split's test set
        sw_score_dict = {sw_score_name: sw_score[test]} if sw_score_name is not None else {}
        baseline_scores.append(scoring(model, X.values[test], y.values[test], **sw_score_dict, **score_params))

        # Get scores for permuted features
        for j, cluster in enumerate(clusters):
            X_test = X.iloc[test, :].copy()

            for _ in range(n_permutations):
                # Permute all features in the cluster
                for col in cluster['columns']:
                    rs.shuffle(X_test[col].values)

                ft_score = scoring(model, X_test, y.values[test], **sw_score_dict, **score_params)
                feature_scores[j].append(baseline_scores[i] - ft_score)

    importance = pd.DataFrame(np.array(feature_scores).T, columns=[c['name'] for c in clusters])
    if raw:
        return importance

    return pd.concat({'mean': importance.mean(), 'std': importance.std()}, axis=1)
