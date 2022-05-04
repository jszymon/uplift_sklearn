# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:19:24 2022

@author: Krzysztof Rudaś
"""

"""Uplift models based on multiple classification/regression models."""

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork
from scipy.linalg.misc import LinAlgError, LinAlgWarning
from scipy import linalg
from scipy.linalg import lstsq
import scipy.sparse as sp
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_X_y, check_consistent_length
from sklearn.neural_network import MLPRegressor
from sklearn.utils.metaestimators import _BaseComposition
from ..base import UpliftRegressorMixin
from ..utils import check_trt

class MLPUpliftRegressor(_BaseComposition, UpliftRegressorMixin):
    """Multimodel uplift regressor.

    Build separate models for control and all treatments, subtract
    control predictions from treatment predictions.

    Parameters
    ----------

    base_estimator : a sklearn regressor or list of (name, regressor)
        tuples.  If a list is provided the first model is used for
        control, successive one for treatments.  If different parameters are
        to be used per each model, the list version must be given.  If a
        single estimator is given it will be cloned for every treatment.
    """
    def __init__(self, base_estimator=MLPRegressor(hidden_layer_sizes=(20,20),max_iter = 2000)):
        self.base_estimator = base_estimator
    def _check_base_estimator(self, n_models):
        if hasattr(self.base_estimator, "fit"):
            estimator_list = []
            for i in range(n_models):
                name = "model_c" if i == 0 else "model_t" + str(i-1)
                estimator_list.append((name, clone(self.base_estimator)))
        else:
            estimator_list = self.base_estimator
        return estimator_list
    def fit(self, X, y, trt, n_trt=None):
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.trt_, self.n_trt_ = check_trt(trt, n_trt)
        check_consistent_length(X, y, self.trt_)
        self.n_models_ = self.n_trt_ + 1
        self.models_ = self._check_base_estimator(self.n_models_)
        #import pdb; pdb.set_trace()
        for i in range(self.n_models_):
            mi = self.models_[i][1]
            ind = (trt==i)
            Xi = X[ind]
            yi = y[ind]
            mi.fit(Xi, yi)
        return self
    def predict(self, X):
        y_control = self.models_[0][1].predict(X)
        cols = [self.models_[i+1][1].predict(X) - y_control
                    for i in range(self.n_trt_)]
        if self.n_trt_ == 1:
            y = cols[0]
        else:
            y = np.column_stack(cols)
        return y
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        if hasattr(self.base_estimator, "fit"):
            return super().get_params(deep=deep)
        return self._get_params('base_estimator', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        if hasattr(self.base_estimator, "fit"):
            return super().set_params(**kwargs)
        else:
            self._set_params('base_estimator', **kwargs)
        return self
