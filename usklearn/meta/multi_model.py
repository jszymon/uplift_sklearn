"""Uplift models based on multiple classification/regression models."""

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork
from scipy.linalg.misc import LinAlgError, _datacopied, LinAlgWarning
from scipy import linalg
from scipy.linalg import lstsq
import scipy.sparse as sp
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_X_y, check_consistent_length
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel
from sklearn.utils.metaestimators import _BaseComposition
from scipy.linalg.decomp import _asarray_validated
from ..base import UpliftRegressorMixin
from ..utils import check_trt

class MultimodelUpliftRegressor(_BaseComposition, UpliftRegressorMixin):
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
    def __init__(self, base_estimator=LinearRegression()):
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
        self.n_ = np.empty(self.n_models_, dtype=int)
        self.p = X.shape[1]
        self.sigma = np.empty(self.n_models_)
        #import pdb; pdb.set_trace()
        for i in range(self.n_models_):
            mi = self.models_[i][1]
            ind = (trt==i)
            self.n_[i] = ind.sum()
            Xi = X[ind]
            yi = y[ind]
            mi.fit(Xi, yi)
            resid =  yi-mi.predict(Xi)
            self.sigma[i] = np.sqrt(np.sum(resid**2)/(len(yi)-Xi.shape[1]))
            Xi1 = np.column_stack((np.ones(self.n_[i]), Xi))
            Xi_2 = Xi1.T@Xi1
            if(i==0):
                X_2 = [Xi_2]
            else:
                X_2.append(Xi_2)
        self.X_2 = np.array(X_2)
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


class MultimodelUpliftLinearRegressor(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ui = self.models_[i+1][1].coef_ - c0
            #print(c0)
            ii = self.models_[i+1][1].intercept_ - i0
            coef[i,:] = ui
            intercept[i] = ii
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)

class MultimodelUpliftLinearRegressorJamesSeparate(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_
        coef_all0 = np.concatenate((np.array([i0]),c0))
        alpha0 = (1-(self.p-3)*self.sigma[0]**2/((coef_all0-np.mean(coef_all0))@self.X_2[0]@(coef_all0-np.mean(coef_all0))))
        
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ci = self.models_[i+1][1].coef_
            ii2 = self.models_[i+1][1].intercept_
            coef_alli = np.concatenate((np.array([ii2]),ci))
            alphai = (1-(self.p-3)*self.sigma[i+1]**2/((coef_alli-np.mean(coef_alli))@self.X_2[i+1]@(coef_alli-np.mean(coef_alli))))
            ui = alphai*ci - alpha0*c0
            #print(c0)
            ii = alphai*ii2 - alpha0*i0
            coef[i,:] = ui
            intercept[i] = ii
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)

class MultimodelUpliftLinearRegressorJamesU(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_

        
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ci = self.models_[i+1][1].coef_
            ii2 = self.models_[i+1][1].intercept_
            ui = ci - c0
            #print(c0)
            ii = ii2 - i0
            coef_all = np.concatenate((np.array([ii]),ui))
            alpha = (1-(self.p-3)/((coef_all -np.mean(coef_all))@np.linalg.pinv(self.sigma[i+1]**2*np.linalg.pinv(self.X_2[i+1])+self.sigma[0]**2*np.linalg.pinv(self.X_2[0]))@(coef_all-np.mean(coef_all))))        
            coef[i,:] = alpha*ui
            intercept[i] = alpha*ii
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)


class MultimodelUpliftLinearRegressorMSESeparate(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_
        var_beta0 = self.sigma[0]**2*np.linalg.pinv(self.X_2[0])
        coef_all0 = np.concatenate((np.array([i0]),c0))
        W0 = self.X_2[0]/self.X_2[0].shape[0]
        alpha0 = coef_all0.T@W0@coef_all0/(coef_all0.T@W0@coef_all0+np.trace(W0@var_beta0))
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ci = self.models_[i+1][1].coef_
            ii2 = self.models_[i+1][1].intercept_
            var_betai = self.sigma[i+1]**2*np.linalg.pinv(self.X_2[i+1])
            coef_alli = np.concatenate((np.array([ii2]),ci))
            Wi = self.X_2[i+1]/self.X_2[i+1].shape[0]
            alphai = coef_alli.T@Wi@coef_alli/(coef_alli.T@Wi@coef_alli+np.trace(Wi@var_betai))
            ui = alphai*ci - alpha0*c0
            #print(c0)
            ii = alphai*ii2 - alpha0*i0
            coef[i,:] = ui
            intercept[i] = ii
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)
    
    
class MultimodelUpliftLinearRegressorMSEU(MultimodelUpliftRegressor, LinearModel):
    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        self._set_coef()
        return self
    def _set_coef(self):
        if not hasattr(self.models_[0][1], "coef_"):
            raise RuntimeError("Base estimator for multi-linear"
                                   " model must set coef_ attribute")
        c0 = self.models_[0][1].coef_
        i0 = self.models_[0][1].intercept_
        var_beta0 = self.sigma[0]**2*np.linalg.pinv(self.X_2[0])
        coef_all0 = np.concatenate((np.array([i0]),c0))
        
        coef = np.empty((self.n_trt_, len(c0)))
        intercept = np.empty(self.n_trt_)
        for i in range(self.n_trt_):
            ci = self.models_[i+1][1].coef_
            ii2 = self.models_[i+1][1].intercept_
            var_betai = self.sigma[i+1]**2*np.linalg.pinv(self.X_2[i+1])
            coef_alli = np.concatenate((np.array([ii2]),ci))
            ui = ci - c0
            #print(c0)
            ii = ii2 - i0
            coef_all = np.concatenate((np.array([ii]),ui))
            W = (self.X_2[0]+self.X_2[i+1])/(self.n_[0]+self.n_[i+1])
            A = np.column_stack((np.vstack((coef_alli.T@W@coef_alli+np.trace(W@var_betai),-coef_all0.T@W@coef_alli)),
                         np.vstack((-coef_alli.T@W@coef_all0,coef_all0.T@W@coef_all0+np.trace(W@var_beta0)))))
    
            b = np.array([coef_alli.T@W@coef_all,-coef_all0.T@W@coef_all]) 
            alphas = np.linalg.solve(A,b)
    
            alpha_t = alphas[0]
            alpha_c = alphas[1]
            
            coef[i,:] = alpha_t*ci - alpha_c*c0
            intercept[i] = alpha_t*ii2 - alpha_c*i0
        if self.n_trt_ == 1:
            coef = np.squeeze(coef)
            intercept = np.squeeze(intercept)
        self.coef_ = coef
        self.intercept_ = intercept
    def predict(self, X):
        return LinearModel.predict(self, X)    
    