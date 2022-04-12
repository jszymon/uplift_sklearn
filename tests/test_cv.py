import numpy as np

from usklearn.utils.multi_array import MultiArray

def test_multiarray1():
    n = 100
    p = 10
    X = np.repeat(np.arange(n, dtype=float).reshape(-1,1), p)
    y = np.arange(n)
    w = np.arange(n, dtype=float)
    trt = np.arange(n) % 3
    n_trt = 3
    m = MultiArray(X, {"w":w, "trt":trt}, {"n_trt":n_trt})
    m[1]
    m[50:100]
