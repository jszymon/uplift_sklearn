"""
=========
Hillstrom
=========

Build uplift classification and regression models on Hillstrom data.
"""

import numpy as np

from sklearn.base import is_classifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from usklearn.datasets import fetch_Hillstrom
from usklearn.meta import MultimodelUpliftRegressor
from usklearn.meta import MultimodelUpliftClassifier
from usklearn.metrics import uplift_curve
from usklearn.model_selection import cross_validate, cross_val_score, uplift_check_cv

import matplotlib.pyplot as plt


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
n_trt = 1

base_classifier = Pipeline([("scaler", StandardScaler()),
                            ("logistic", LogisticRegression(max_iter=100))])
models = [MultimodelUpliftRegressor(),
          MultimodelUpliftClassifier(base_estimator=base_classifier)]
cv, y_stratify = uplift_check_cv(StratifiedShuffleSplit(test_size=10000, random_state=123),
                                 y, trt, n_trt, classifier=True)
colors = "rgbk"
for train_index, test_index in cv.split(X, y_stratify):
    print((y[test_index]).sum())
    for mi, m in enumerate(models):
        m.fit(X[train_index], y[train_index], trt[train_index], n_trt)
        score = m.predict(X[test_index])
        if is_classifier(m):
            score = score[:,1]
        x, u = uplift_curve(y[test_index], score, trt[test_index], n_trt)
        plt.plot(x, u, color=colors[mi], alpha=0.3)


plt.show()
