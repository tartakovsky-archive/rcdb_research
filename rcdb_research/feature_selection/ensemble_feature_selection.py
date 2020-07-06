from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted





class EFS(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    def __init__(self,
                 estimator: BaseEstimator):
        pass

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        pass

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        pass

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        pass

    def transform(self, X):
        # TODO: Override to support clusters and X: pd.DataFrame
        pass

    def inverse_transform(self, X):
        # TODO: Override to support clusters and X: pd.DataFrame
        pass

    def _get_support_mask(self):
        pass
