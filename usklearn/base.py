"""Base classes for all estimators and trasformers."""

from sklearn.base import BaseEstimator

from .metrics import e_sate

class UpliftRegressorMixin(object):
    def score(self, X, y, trt, sample_weight=None):
        return e_sate(y, self.predict(X), trt, n_trt=self.n_trt_)
