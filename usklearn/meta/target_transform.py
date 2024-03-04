"""Uplift models based on target transform."""

import numpy as np

from sklearn.utils import check_X_y, check_consistent_length
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from ..utils import check_trt
from .multi_model import MultimodelUpliftRegressor
from .multi_model import _MultimodelUpliftClassifierBase


class TargetTransformUpliftRegressor(MultimodelUpliftRegressor):
    def __init__(self, base_estimator=LinearRegression()):
        super().__init__(base_estimator=base_estimator,
                         ignore_control=True)
    def fit(self, X, y, trt, n_trt=None):
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.trt_, self.n_trt_ = check_trt(trt, n_trt)
        check_consistent_length(X, y, self.trt_)
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
            n_ti = t_mask.sum()
            n = n_c + n_ti
            self.n_[i] = n
            Xi = X[mask]
            yi = np.asfarray(y[mask]) # allow classification problems
            yi[c_mask[mask]] *= (-n/n_c)
            yi[t_mask[mask]] *= (n/n_ti)
            mi.fit(Xi, yi)
        return self

class TargetTransformUpliftClassifier(_MultimodelUpliftClassifierBase):
    def __init__(self, base_estimator=LogisticRegression()):
        super().__init__(base_estimator=base_estimator,
                         ignore_control=True)
    def fit(self, X, y, trt, n_trt=None):
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.trt_, self.n_trt_ = check_trt(trt, n_trt)
        check_consistent_length(X, y, self.trt_)
        self._set_fit_params(y, trt, n_trt)

        if self.n_classes_ > 2:
            raise RuntimeError("TargetTransformUpliftClassifier "
                               "only supports binary targets.")

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
            n = mask.sum()
            self.n_[i] = n
            Xi = X[mask]
            yi = y[mask]
            yi[c_mask[mask]] = 1 - yi[c_mask[mask]]
            mi.fit(Xi, yi)
        return self
