"""Uplift models based on multiple classification/regression models."""

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_X_y, check_consistent_length, column_or_1d
from sklearn.linear_model import LinearRegression

from .base import UpliftRegressorMixin

class MultimodelUpliftRegressor(BaseEstimator, UpliftRegressorMixin):
    def __init__(self, base_model=LinearRegression()):
        self.base_model = base_model
    def fit(self, X, y, trt, n_trt=None):
        # TODO: check_X_y_trt
        X, y = check_X_y(X, y, accept_sparse="csr")
        trt = column_or_1d(trt)
        check_consistent_length(X, y, trt)
        if not np.issubdtype(trt.dtype, np.integer):
            raise ValueError("Treatment values must be integers")
        if (trt < 0).any():
            raise ValueError("Treatment values must be >= 0")
        # TODO: process_trt
        if n_trt is not None:
            self.n_trt_ = n_trt
            assert max(trt) <= self.n_trt_
        else:
            self.n_trt_ = max(trt)
        self.n_models_ = self.n_trt_ + 1
        self.models_ = []
        for i in range(self.n_models_):
            mi = clone(self.base_model)
            ind = (trt==i)
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
    def predict_action(self, X):
        """Predict most beneficial action."""
        y = self.predict(X)
        if self.n_trt_ == 1:
            a = (y > 0)*1
        else:
            a = np.argmax(y, axis=1) + 1
            best_y = np.max(y, axis=1)
            a[best_y <= 0] == 0
        return a
