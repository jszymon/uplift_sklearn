"""Meta models: uplift models combined of classifiers/regressors."""

from .multi_model import MultimodelUpliftRegressor
from .multi_model import MultimodelUpliftLinearRegressor
from .multi_model import MultimodelUpliftClassifier

__all__ = ["MultimodelUpliftRegressor",
           "MultimodelUpliftLinearRegressor",
           "MultimodelUpliftClassifier"]
