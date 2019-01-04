"""Uplift models based on multiple classification/regression models."""

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_X_y, check_consistent_length, column_or_1d
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.base import LinearModel

from .base import UpliftRegressorMixin
from .utils import check_trt

class MultimodelUpliftRegressor(BaseEstimator, UpliftRegressorMixin):
    def __init__(self, base_estimator=LinearRegression()):
        self.base_estimator = base_estimator
    def fit(self, X, y, trt, n_trt=None):
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.trt_, self.n_trt_ = check_trt(trt, n_trt)
        check_consistent_length(X, y, self.trt_)
        self.n_models_ = self.n_trt_ + 1
        self.models_ = []
        self.n_ = np.empty(self.n_models_, dtype=int)
        for i in range(self.n_models_):
            mi = clone(self.base_estimator)
            ind = (trt==i)
            self.n_[i] = ind.sum()
            Xi = X[ind]
            yi = y[ind]
            mi.fit(Xi, yi)
            self.models_.append(mi)
        return self
    def predict(self, X):
        y_control = self.models_[0].predict(X)
        cols = [self.models_[i+1].predict(X) - y_control
                    for i in range(self.n_trt_)]
        if self.n_trt_ == 1:
            y = cols[0]
        else:
            y = np.column_stack(cols)
        return y


class MultimodelUpliftLinearRegressor(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0].coef_
        i0 = self.models_[0].intercept_
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ui = self.models_[i+1].coef_ - c0
            ii = self.models_[i+1].intercept_ - i0
            coef[i,:] = ui
            intercept[i] = ii
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)
