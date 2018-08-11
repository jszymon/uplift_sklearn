"""Base classes for all estimators and trasformers."""

from sklearn.base import BaseEstimator

class UpliftRegressorMixin(object):
    def score(X, y, trt, sample_weight=None):
        pass
