"""A memoized classifier class.

Used to avoid recomputing the same classifier twice e.g. when both
T-learner and Response models are computed.

"""

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.utils.validation import check_memory

def _do_fit(estimator, model_params, *args, **kwargs):
    return clone(estimator).fit(*args, **kwargs)

class MemoizedClassifier(BaseEstimator):
    def __init__(self, estimator, memory):
        self.estimator = estimator
        self.memory = memory
    def fit(self, *args, **kwargs):
        self.memory_ = check_memory(self.memory)
        if not hasattr(self, "do_fit_cached_"):
            self.do_fit_cached_ = self.memory_.cache(_do_fit, ignore=["estimator"])
        self.fitted_etimator_ = self.do_fit_cached_(self.estimator,
                                                    self.estimator.get_params(),
                                                    *args, **kwargs)
        return self
    
    def __getattr__(self, name):
        if name in ["fitted_etimator_", "do_fit_cached_"]:
            try:
                return self.__dict__[name]
            except:
                raise AttributeError(f"MemoizedClassifier has no attribute {name}")
        return getattr(self.fitted_etimator_, name)
    
