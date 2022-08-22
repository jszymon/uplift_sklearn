"""Copied and modified from sklearn"""

"""
The :mod:`sklearn.model_selection._search` includes utilities to fine-tune the
parameters of an estimator.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

from collections.abc import Mapping, Sequence, Iterable
from functools import partial, reduce
from itertools import product
import numbers

import numpy as np

from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.utils.validation import indexable, check_is_fitted
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.metrics import check_scoring

from ..metrics import check_uplift_scoring

from ._validation import uplift_check_cv
from ._validation import _check_multimetric_scoring


__all__ = ['GridSearchCV', 'ParameterGrid', 'fit_grid_point',
           'ParameterSampler', 'RandomizedSearchCV']

from collections.abc import Mapping, Sequence, Iterable

from sklearn.model_selection import GridSearchCV as _sklearn_GridSearchCV

from ..utils import check_trt
from ..utils import MultiArray

from ._validation import _WrappedUpliftEstimator, _WrappedScoring


class BaseSearchCV(BaseEstimator):
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    def __getattr__(self, arg):
        """Delegate to real grid search."""
        if "wrapped_search_" in self.__dict__:
            return getattr(self.wrapped_search_, arg)
        raise AttributeError(arg)
    def fit(self, X, y, trt, n_trt=None, *, groups=None, **fit_params):
        X, y, trt, groups = indexable(X, y, trt, groups)
        trt, n_trt = check_trt(trt, n_trt)

        wrapped_est = _WrappedUpliftEstimator(self.estimator)
        scoring = check_uplift_scoring(self.estimator, self.scoring)
        wrapped_scoring = _WrappedScoring(scoring)
        # TODO: multimetric scoring
        #scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)
        #wrapped_scorers = {name:_WrappedScoring(scoring) for name, scoring in scorers.items()}
        # TODO: fix cv
        cv, y_stratify = uplift_check_cv(self.cv, y, trt, n_trt, classifier=is_classifier(self.estimator))
        # fix param_grid for wrapped uplift estimator
        param_grid = self.param_grid
        if isinstance(self.param_grid, Mapping):
            param_grid = [param_grid]
        new_param_grid = []
        for grid in param_grid:
            new_grid = {"base_estimator__" + key:value for key, value in grid.items()}
            new_param_grid.append(new_grid)
        self.wrapped_search_ = self.wrapped_search_class(estimator=wrapped_est, param_grid=new_param_grid, scoring=wrapped_scoring,
                                                    cv=cv, n_jobs=self.n_jobs, refit=self.refit, verbose=self.verbose,
                                                    pre_dispatch=self.pre_dispatch, error_score=self.error_score,
                                                        return_train_score=self.return_train_score)

        # multiarray to pass additional data
        # y_stratify is used only for stratification
        Xm = MultiArray(X, array_dict={"y":y, "trt":trt}, scalar_dict={"n_trt":n_trt})
        self.wrapped_search_.fit(Xm, y_stratify, groups=groups, **fit_params)
    def score(self, X, y, trt, n_trt=None):
        Xm = MultiArray(X, array_dict={"y":y, "trt":trt}, scalar_dict={"n_trt":n_trt})
        return self.wrapped_search_.score(Xm, y)

class GridSearchCV(BaseSearchCV):
    wrapped_search_class = _sklearn_GridSearchCV
    def __init__(self, estimator, param_grid, *, scoring=None,
                 n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score




# TODO:
class RandomizedSearchCV(BaseEstimator):
    _required_parameters = ["estimator", "param_distributions"]

    def __init__(self, estimator, param_distributions, *, n_iter=10, scoring=None,
                 n_jobs=None, refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score=np.nan,
                 return_train_score=False):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        evaluate_candidates(ParameterSampler(
            self.param_distributions, self.n_iter,
            random_state=self.random_state))
