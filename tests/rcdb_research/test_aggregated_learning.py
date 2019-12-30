import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from rcdb_research.cross_validation.aggregated_learning import \
    MultiInputSplitter, aggregate_splits, predict_aggregated_splits


X = pd.DataFrame(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    columns=['X'],
    index=pd.date_range('2019-11-15', periods=11)
)
y = pd.Series(
    [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    index=pd.date_range('2019-11-15', periods=11)
)

X1 = pd.DataFrame(
    -np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    columns=['X1'],
    index=pd.DatetimeIndex(['2019-11-14', '2019-11-15', '2019-11-17', '2019-11-17', '2019-11-18',
                            '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22',
                            '2019-11-23', '2019-11-24', '2019-11-25'])
)
y1 = pd.Series(
    -np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0]),
    index=pd.DatetimeIndex(['2019-11-14', '2019-11-15', '2019-11-17', '2019-11-17', '2019-11-18',
                            '2019-11-19', '2019-11-20', '2019-11-21', '2019-11-22',
                            '2019-11-23', '2019-11-24', '2019-11-25'])
)


@pytest.mark.parametrize(
    'drop_earlier_history, empty_additional',
    [
        [True, False],
        [True, True],
        [False, True],
        [False, False],
    ]
)
def test_MultiInputSplitter_split(drop_earlier_history, empty_additional):
    print('drop_earlier_history', drop_earlier_history)

    if not empty_additional:
        splits = MultiInputSplitter(3, train_size=0.75, drop_earlier_history=drop_earlier_history).split(
            main=(X, y),
            additional=[(X1, y1)],
            verbose=True,
        )
    else:
        splits = MultiInputSplitter(3, train_size=0.75, drop_earlier_history=drop_earlier_history).split(
            main=(X, y),
            verbose=True,
        )

    assert len(splits) == 3
    for split in splits:
        X_test = split['main']['X_test']
        y_test = split['main']['y_test']

        assert y_test.index.equals(X_test.index)

        main_train = [
            dict(
                X_train=split['main']['X_train'],
                y_train=split['main']['y_train']
            )
        ]
        for additional in split['additional'] + main_train:
            X_train = additional['X_train']
            y_train = additional['y_train']

            assert y_train.index.equals(X_train.index)

            assert (X_train.index < X_test.index[0]).all()

            if not empty_additional:
                if drop_earlier_history:
                    assert X.index[0] <= X_train.index[0]
                else:
                    assert X1.index[0] <= X_train.index[0]


@pytest.mark.parametrize(
    'empty_additional', [True, False]
)
def test_aggregate_splits(empty_additional):
    if not empty_additional:
        splits = MultiInputSplitter(3, train_size=0.75).split(
            main=(X, y),
            additional=[(X1, y1)],
        )
    else:
        splits = MultiInputSplitter(3, train_size=0.75).split(
            main=(X, y),
        )
    res = aggregate_splits(
        splits=splits, pre_agg_transforms=MinMaxScaler()
    )
    assert len(res) == 3
    for agg_split, split in zip(res, splits):
        assert len(agg_split) == 4
        assert agg_split[0].size == \
            split['main']['X_train'].values.size + sum(x['X_train'].size for x in split['additional'])
        assert agg_split[1].size == \
            split['main']['y_train'].values.size + sum(x['y_train'].size for x in split['additional'])

        assert agg_split[2].size == split['main']['X_test'].values.size
        assert (agg_split[3] == split['main']['y_test'].values).all()


@pytest.mark.parametrize(
    'n_jobs', [1, -1, 2]
)
def test_predict_aggregated_splits(n_jobs):
    splits = MultiInputSplitter(3, train_size=0.75).split(
        main=(X, y),
        additional=[(X1, y1)],
    )
    agg_splits = aggregate_splits(
        splits=splits, pre_agg_transforms=MinMaxScaler()
    )
    res = predict_aggregated_splits(
        clf=DecisionTreeClassifier(),
        agg_splits=agg_splits,
        n_jobs=n_jobs
    )
    assert len(res) == 2
    assert res[0].shape[0] == res[1].shape[0]
