"""Implement scorers for uplift modeling."""

from sklearn.metrics._scorer import _BaseScorer

class _UpliftPredictScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, trt, n_trt=None, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_pred = method_caller(estimator, "predict", X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred, trt, n_trt,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, trt, n_trt,
                                                 **self._kwargs)
class _UpliftDecisionScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, trt, n_trt=None, sample_weight=None):
        """Evaluate predicted treatment decisions for X relative to y_true.

        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.

        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        a = method_caller(estimator, "predict_action", X)
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, a, trt, n_trt,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, a, trt, n_trt,
                                                 **self._kwargs)

def make_uplift_scorer(score_func, greater_is_better=True, needs_decision=False,
                       needs_proba=False, needs_threshold=False, **kwargs):
    """Make a scorer from a performance metric or loss function.
    This factory function wraps scoring functions for use in GridSearchCV
    and cross_val_score. It takes a score function, such as ``accuracy_score``,
    ``mean_squared_error``, ``adjusted_rand_index`` or ``average_precision``
    and returns a callable that scores an estimator's output.
    Read more in the :ref:`User Guide <scoring>`.
    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.
        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.
    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    """
    if needs_proba or needs_threshold:
        raise NotImplementedError()
    sign = 1 if greater_is_better else -1
    if needs_proba + needs_decision + needs_threshold > 1:
        raise ValueError("Set only one of needs_proba, needs_threshold or"
                         " needs_decision to True.")
    if needs_proba:
        cls = _UpliftProbaScorer
    elif needs_threshold:
        cls = _UpliftThresholdScorer
    elif needs_decision:
        cls = _UpliftDecisionScorer
    else:
        cls = _UpliftPredictScorer
    return cls(score_func, sign, kwargs)
