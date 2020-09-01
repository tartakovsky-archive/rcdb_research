from sklearn.ensemble import RandomForestClassifier

from rcdb_research.feature_selection import SelectKBest, EFS
from rcdb_research.feature_importance import MDI


def test_EFS(Xy):
    X, y = Xy
    efs = EFS(estimators=[MDI(RandomForestClassifier())])
    res = efs.fit_transform(X, y)
    assert res.shape[0] == len(X)
    assert res.shape[1] <= len(X.columns)


def test_SelectKBest(Xy):
    X, y = Xy
    k = 5
    kbest = SelectKBest(MDI(RandomForestClassifier()), k)
    assert kbest.fit_transform(X, y).shape == (len(X), k)
