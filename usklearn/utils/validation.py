"""Utilities for input validation."""

from sklearn.utils import column_or_1d, check_consistent_length

def check_treatment(trt, n_trt, y=None):
    trt = column_or_1d(trt)
    return trt, n_trt
