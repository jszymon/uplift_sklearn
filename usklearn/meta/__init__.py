"""Meta models: uplift models combined of classifiers/regressors."""

from .multi_model import MultimodelUpliftRegressor
from .multi_model import MultimodelUpliftLinearRegressor
from .multi_model import MultimodelUpliftClassifier
from .response import TreatmentUpliftClassifier
from .response import ResponseUpliftClassifier
from .response import ControlUpliftClassifier
from .target_transform import TargetTransformUpliftRegressor
from .target_transform import TargetTransformUpliftClassifier
from .s_learner import SLearnerUpliftRegressor
from .s_learner import SLearnerUpliftClassifier
from .nested import NestedMeanUpliftRegressor
from .nested import DDRUpliftClassifier

__all__ = ["MultimodelUpliftRegressor",
           "MultimodelUpliftLinearRegressor",
           "MultimodelUpliftClassifier",
           "TreatmentUpliftClassifier",
           "ResponseUpliftClassifier",
           "ControlUpliftClassifier",
           "TargetTransformUpliftRegressor",
           "TargetTransformUpliftClassifier",
           "SLearnerUpliftRegressor",
           "SLearnerUpliftClassifier",
           "NestedMeanUpliftRegressor",
           "DDRUpliftClassifier",
           ]
