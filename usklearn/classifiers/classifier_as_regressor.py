"""Use a classifier as a regressor.

By default return predicted probabilities as numeric predictions.

"""


from sklearn.base import BaseEstimator, RegressorMixin

class ClassifierAsRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, response_method='predict_proba',
                 pos_label=1):
        """Wraps a classifier such that it behaves like a regressor.

        The predict method return by default predicted probability for
        class specified by pos_label (default 1).  The method used for
        prediction can be changed by passing the response_method
        argument.

        If response_method return a vector (e.g. decision_function)
        pos_label will be ignored.

        """
        self.estimator = estimator
        self.response_method = response_method
        self.pos_label = pos_label
    def fit(self, *args, **kwargs):
        self.fitted_estimator_ = self.estimator.fit(*args, **kwargs)
        return self
    def predict(self, *args, **kwargs):
        resp_method = getattr(self.fitted_estimator_, self.response_method)
        preds = resp_method(*args, **kwargs)
        pred_ndim = len(preds.shape)
        if pred_ndim > 2:
            raise RuntimeError("ClassifierAsRegressor: response method"
                               " must return a vector of a matrix.")
        elif pred_ndim == 2:
            preds = preds[:,self.pos_label]
        return preds

    def __getattr__(self, name):
        if name in ["fitted_estimator_", "response_method", "pos_label"]:
            try:
                return self.__dict__[name]
            except:
                raise AttributeError(f"ClassifierAsRegressor has"
                                     " no attribute {name}")
        if "fitted_estimator_" not in self.__dict__:
            return getattr(self.estimator, name)
        return getattr(self.fitted_estimator_, name)
