"""Meta models: uplift models combined of classifiers/regressors."""

from .multi_model import MultimodelUpliftRegressor
from .multi_model import MultimodelUpliftLinearRegressor

__all__ = ["MultimodelUpliftRegressor",
           "MultimodelUpliftLinearRegressor"]
