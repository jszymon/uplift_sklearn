import numpy as np

from sklearn.datasets import fetch_openml

def fetch_Orange_churn():
    D = fetch_openml("churn-uplift-orange", parser="auto")
    D.trt = np.asarray(D.data.t, dtype=np.int32)
    D.data = D.data.drop(columns="t")
    return D
