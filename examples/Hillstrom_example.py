"""
=========
Hillstrom
=========

Build uplift classification and regression models on Hillstrom data.
"""

import numpy as np

from usklearn.datasets import fetch_Hillstrom

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
print(X)
