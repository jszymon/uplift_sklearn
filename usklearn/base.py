"""Base classes for all estimators and trasformers."""

from sklearn.base import BaseEstimator

from .metrics import e_sate

class UpliftRegressorMixin(object):
    _estimator_type = "regressor"
    _uplift_model = True

    def score(self, X, y, trt, sample_weight=None):
        return e_sate(y, self.predict(X), trt, n_trt=self.n_trt_)

def is_uplift(estimator):
    """Returns True if the given estimator is an uplift model.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an uplift model and False otherwise.
    """
    return getattr(estimator, "_uplift_model", None) is True
