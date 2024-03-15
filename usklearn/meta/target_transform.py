"""Uplift models based on target transform."""

import numpy as np

from sklearn.utils import check_X_y, check_consistent_length
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from ..utils import check_trt
from .multi_model import MultimodelUpliftRegressor
from .multi_model import _MultimodelUpliftClassifierBase

class _TargetTransformUpliftModelBase:
    def fit(self, X, y, trt, n_trt=None, sample_weight=None):
        """Generic fitting method to be called """
        X, y = check_X_y(X, y, accept_sparse="csr")
        trt, n_trt = check_trt(trt, n_trt)
        check_consistent_length(X, y, trt)
        self._set_fit_params(y, trt, n_trt)

        self.n_models_ = self.n_trt_ + 1
        self.models_ = self._check_base_estimator(self.n_models_)
        self.n_ = np.empty(self.n_models_, dtype=int)
        c_mask = (trt==0)
        n_c = c_mask.sum()

        for i in range(self.n_models_):
            if self.ignore_control and i == 0:
                continue
            mi = self.models_[i][1]
            t_mask = (trt==i)
            mask = t_mask|c_mask
            if sample_weight is None:
                wi = None
                n = mask.sum()
            else:
                wi = sample_weight[mask]
                n = wi.sum()
            self.n_[i] = n
            Xi = X[mask]
            yi = np.asfarray(y[mask]) # allow classification problems
            trt_i = trt[mask]
            Xi, yi, wi = self._transform(Xi, yi, trt_i, n_trt, wi, y)
            if wi is None:
                mi.fit(Xi, yi)
            else:
                mi.fit(Xi, yi, sample_weight=wi)
        return self
    def _transform(self, X, y, trt, n_trt, sample_weight, full_y):
        """Transform target for model building.

        full_y is passed to allow tests to avoid overwriting."""
        raise NotImplementedError()

class TargetTransformUpliftRegressor(_TargetTransformUpliftModelBase, MultimodelUpliftRegressor):
    def __init__(self, base_estimator=LinearRegression()):
        super().__init__(base_estimator=base_estimator,
                         ignore_control=True)
    #def fit(self, X, y, trt, n_trt=None, sample_weight=None):
    #    return self.transformed_fit(X, y, trt, n_trt=n_trt, sample_weight=sample_weight)
    def _transform(self, X, y, trt, n_trt, sample_weight, full_y):
        """Transform target for model building.

        full_y is passed to allow tests to avoid overwriting."""
        n = trt.shape[0]
        mask_c = (trt==0)
        mask_t = ~mask_c
        nt = mask_t.sum()
        nc = mask_c.sum()
        n = nt + nc
        y = np.asfarray(y) # allow classification problems
        if np.may_share_memory(y, full_y):
            y = y.copy()
        y[mask_c] *= (-n/nc)
        y[mask_t] *= (n/nt)
        return X, y, sample_weight


class TargetTransformUpliftClassifier(_TargetTransformUpliftModelBase, _MultimodelUpliftClassifierBase):
    def __init__(self, base_estimator=LogisticRegression()):
        super().__init__(base_estimator=base_estimator,
                         ignore_control=True)
    #def fit(self, X, y, trt, n_trt=None, sample_weight=None):
    #    return self.transformed_fit(X, y, trt, n_trt=n_trt, sample_weight=sample_weight)
    def _transform(self, X, y, trt, n_trt, sample_weight, full_y):
        """Transform target for model building.

        full_y is passed to allow tests to avoid overwriting."""
        if np.may_share_memory(y, full_y):
            y = y.copy()
        y[trt == 0] = 1-y[trt == 0]
        return X, y, sample_weight
    def predict(self, X):
        y = super().predict(X)
        y = 2*y-1
        return y
