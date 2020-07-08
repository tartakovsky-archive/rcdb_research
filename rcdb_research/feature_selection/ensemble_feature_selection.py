# from typing import Optional, Union, Any
#
# from sklearn.base import BaseEstimator, MetaEstimatorMixin
# from sklearn.feature_selection import SelectorMixin
# from sklearn.utils.metaestimators import if_delegate_has_method
# from sklearn.utils.validation import check_is_fitted
#
# from sklearn.feature_selection import SelectFromModel

# class EFS(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
#     def __init__(self,
#                  selectors: list):
#         pass
#
#     def fit(self, X, y, clusters=None, labels=None, **fit_params):
#         # Vote
#         # y_pred_result = np.apply_along_axis(
#         #     lambda x: np.argmax(np.bincount(x, weights=weights)),
#         #     axis=0, arr=y_preds
#         # )
#         pass
#
#     def _get_support_mask(self):
#         pass
