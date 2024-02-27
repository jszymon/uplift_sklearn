"""Meta models: uplift models combined of classifiers/regressors."""

from .multi_model import MultimodelUpliftRegressor
from .multi_model import MultimodelUpliftLinearRegressor
from .multi_model import MultimodelUpliftClassifier
from .multi_model import TreatmentUpliftClassifier
from .multi_model import ResponseUpliftClassifier

__all__ = ["MultimodelUpliftRegressor",
           "MultimodelUpliftLinearRegressor",
           "MultimodelUpliftClassifier",
           "TreatmentUpliftClassifier",
           "ResponseUpliftClassifier"]
