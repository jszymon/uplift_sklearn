"""Base classes for all estimators and trasformers."""

from sklearn.base import BaseEstimator

from .metrics import e_sate

class UpliftRegressorMixin(object):
    _estimator_type = "regressor"
    _uplift_model = True

    def score(self, X, y, trt, sample_weight=None):
        return -e_sate(y, self.predict(X), trt, n_trt=self.n_trt_)
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
