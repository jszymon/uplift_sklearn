"""
=========
Hillstrom
=========

Build uplift classification and regression models on Hillstrom data.
"""

import numpy as np

from sklearn.linear_model import Ridge

from usklearn.datasets import fetch_Hillstrom
from usklearn.meta import MultimodelUpliftRegressor
from usklearn.meta import MultimodelUpliftLinearRegressor
from usklearn.metrics import uplift_curve
from usklearn.model_selection import cross_validate, cross_val_score
from usklearn.model_selection import GridSearchCV

def encode_features(D):
    """Convert features to float matrix.

    Use K-1 encoding for categorical variables."""
    X = D.data
    cols = []
    for c in D.feature_names:
        if c not in D.categ_values:
            cols.append(np.asfarray(X[c]))
        else:
            n_categs = len(D.categ_values[c])
            x = np.eye(n_categs)[X[c]]
            cols.append(x[:,:-1]) # skip last category
    return np.column_stack(cols)

D = fetch_Hillstrom(as_frame=True)
X = encode_features(D)
y = D.target_visit
trt = D.treatment
# keep women's campaign
mask = ~(trt == 1)
X = X[mask]
y = y[mask]
trt = (trt[mask] == 2)*1

# TODO: change to classification
rlin = MultimodelUpliftLinearRegressor()
rlin.fit(X, y, trt)

score = rlin.predict(X)
x, u = uplift_curve(y, score, trt)

import matplotlib.pyplot as plt
plt.plot(x, u)
plt.show()
