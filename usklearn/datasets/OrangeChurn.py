import numpy as np

from sklearn.datasets import fetch_openml

def fetch_Orange_churn(return_X_y=False, **args):
    D = fetch_openml("churn-uplift-orange", parser="auto", **args)
    D.trt = np.asarray(D.data.t, dtype=np.int32)
    D.data = D.data.drop(columns="t")
    D.feature_names.remove("t")
    D.treatment_values = ["0", "1"]
    if return_X_y:
        return D.data, D.target.values.astype(np.int32), D.trt
    return D
