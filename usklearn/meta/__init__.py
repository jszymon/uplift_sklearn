"""Meta models: uplift models combined of classifiers/regressors."""

from .multi_model import MultimodelUpliftRegressor
from .multi_model import MultimodelUpliftLinearRegressor
from .multi_model import MultimodelUpliftClassifier
from .response import TreatmentUpliftClassifier
from .response import ResponseUpliftClassifier
from .response import ControlUpliftClassifier

__all__ = ["MultimodelUpliftRegressor",
           "MultimodelUpliftLinearRegressor",
           "MultimodelUpliftClassifier",
           "TreatmentUpliftClassifier",
           "ResponseUpliftClassifier",
           "ControlUpliftClassifier"]
