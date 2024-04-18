"""Nested models where control outcome predictions are used in an
uplift model."""

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from ..utils import safe_hstack

from .base import UpliftMetaModelBase
from ..base import UpliftRegressorMixin
from ..base import UpliftClassifierMixin

class NestedMeanUpliftRegressor(UpliftMetaModelBase, UpliftRegressorMixin):
    """Nested regression model.

    First builds a model on controls, then subtracts its training
    predictions from target.  An uplift model is then build on the new
    target.

    Only available for regression models.

    """
    def __init__(self, base_estimator = LinearRegression()):
        super().__init__(base_estimator=base_estimator)
    def _get_model_names_list(self, X=None, y=None, trt=None):
        m_names = ["model_c"]
        for i in range(self.n_trt_):
            name = "model_u"
            if self.n_trt_ > 1:
                name += str(i-1)
            m_names.append(name)
        return m_names
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        c_mask = (trt==0)
        y_c = np.asfarray(y[c_mask]) # allow classification problems
        if sample_weight is not None:
            yield X[c_mask], y_c, sample_weight[c_mask]
        else:
            yield X[c_mask], y_c, None
        # assume the control model is already fitted
        m_c = self.models_[0][1]
        for i in range(self.n_trt_):
            t_mask = (trt==(i+1))
            X_i = X[t_mask]
            y_i_pred = m_c.predict(X_i)
            y_i = y[t_mask] - y_i_pred
            if sample_weight is not None:
                w_i = sample_weight[t_mask]
            else:
                w_i = None
            yield X_i, y_i, w_i
    def predict(self, X):
        preds = [m_i.predict(X) for _, m_i in self.models_[1:]]
        if self.n_trt_ == 1:
            y = preds[0]
        else:
            y = np.column_stack(preds)
        return y


class DDRUpliftClassifier(UpliftMetaModelBase, UpliftClassifierMixin):
    """Dependent Data Representation metamodel.  It is a double model
    where control predictions are added as a variable in the treatment
    model.

    The model was proposed in A. Betlei, E. Diemert, and M.-R. Amini
    Uplift Prediction with Dependent Feature Representation in
    Imbalanced Treatment and Control Conditions, ICONIP, 2018.

    """
    def __init__(self, base_estimator=LogisticRegression(),
                 control_prediction_method="predict_proba"):
        super().__init__(base_estimator=base_estimator)
        self.control_prediction_method = control_prediction_method
    def _get_model_names_list(self, X=None, y=None, trt=None):
        m_names = ["model_c"]
        for i in range(self.n_trt_):
            name = "model_t"
            if self.n_trt_ > 1:
                name += str(i-1)
            m_names.append(name)
        return m_names
    def _control_prediction_feature(self, X):
        m_c = self.models_[0][1]
        if self.control_prediction_method == "predict_proba":
            y = m_c.predict_proba(X)[:,1:]
        elif self.control_prediction_method == "predict":
            y = m_c.predict(X)
        elif self.control_prediction_method == "decision_function":
            y = m_c.decision_function(X)
        return y
    def _iter_training_subsets(self, X, y, trt, n_trt, sample_weight):
        c_mask = (trt==0)
        if sample_weight is not None:
            yield X[c_mask], y[c_mask], sample_weight[c_mask]
        else:
            yield X[c_mask], y[c_mask], None
        for i in range(self.n_trt_):
            t_mask = (trt==(i+1))
            X_i = X[t_mask]
            y_i = y[t_mask]
            # assume the control model is already fitted
            y_i_pred = self._control_prediction_feature(X_i)
            X_i = safe_hstack([X_i, y_i_pred])
            if sample_weight is not None:
                w_i = sample_weight[t_mask]
            else:
                w_i = None
            yield X_i, y_i, w_i
    def predict(self, X):
        y_pred = self._control_prediction_feature(X)
        pred_c = self.models_[0][1].predict_proba(X)
        preds = []
        for i in range(self.n_trt_):
            X_i = safe_hstack([X, y_pred])
            pred_i = self.models_[i+1][1].predict_proba(X_i) - pred_c
            preds.append(pred_i)
        if self.n_trt_ == 1:
            y = preds[0]
        else:
            y = np.dstack(preds)
        return y
