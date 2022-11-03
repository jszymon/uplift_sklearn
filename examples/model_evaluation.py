"""Demo of some model selection and evaluation functions."""

import numpy as np

from sklearn.linear_model import Ridge

from usklearn.datasets import fetch_Lalonde
from usklearn.meta import MultimodelUpliftRegressor
from usklearn.model_selection import cross_val_predict

D = fetch_Lalonde("B")
X = D.data
y = D.target
trt = D.treatment

r = MultimodelUpliftRegressor()
print(cross_val_predict(r, X, y, trt, n_trt=1, cv=5))
