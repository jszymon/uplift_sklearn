"""
=========
Hillstrom
=========

Build uplift classification and regression models on Hillstrom data.
"""

import numpy as np

from usklearn.datasets import fetch_Hillstrom
from usklearn.multi_model import MultimodelUpliftRegressor
from usklearn.metrics import e_sate, e_satt
from usklearn.model_selection import cross_validate, cross_val_score

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


D = fetch_Hillstrom()
X = encode_features(D)
y = D.target_spend
trt = D.treatment

r = MultimodelUpliftRegressor()
r.fit(X, y, trt)
print(r)
print("training SATE:", r.score(X, y, trt))
print("training SATT:", e_satt(y, r.predict(X), trt, n_trt=2))
print("crossval SATE:", cross_val_score(r, X, y, trt, n_trt=2, cv=10))
print("crossval SATT:", cross_validate(r, X, y, trt, n_trt=2, cv=10, scoring="e_sate"))

# merge treatments
trt[trt==2] = 1
r.fit(X, y, trt)
print(r)
print("training SATE:", r.score(X, y, trt))
print("training SATT:", e_satt(y, r.predict(X), trt))
print("crossval SATE:", cross_val_score(r, X, y, trt, n_trt=1, cv=10))

