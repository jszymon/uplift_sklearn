"""Uplift and Qini curves."""

import numpy as np

from sklearn.utils.validation import check_array, check_consistent_length

from ..utils import check_trt
from ..utils import area_under_curve

def _cumulative_gains_curve(y_true, y_score, sample_weight):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    #y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.0

    # handle tied values
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # compute gains and prepend (0,0) point at the beginning
    gains = np.r_[0, np.cumsum(y_true * weight)[threshold_idxs]]
    threshold_idxs = np.r_[0, threshold_idxs+1]
    return threshold_idxs, gains

def uplift_curve(y_true, y_score, trt, n_trt=None, pos_label=None, sample_weight=None):
    """Uplift curve.

    Unless specified explicitly, y_true is assumed to be 0-1, with 1
    the positive outcome.

    This function implements the variant used by Rzepakowski and
    Jaroszewicz, where treatment and control curves are computed
    separately and subtracted.

    """
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    trt, n_trt = check_trt(trt, n_trt)
    if sample_weight is None:
        check_consistent_length(y_true, y_score, trt)
        sample_weight_c = None
        sample_weight_t = None
    else:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(y_true, y_score, trt, sample_weight)
        sample_weight_c = sample_weight[trt==0]
        sample_weight_t = sample_weight[trt==t]
    if n_trt > 1:
        raise ValueError("uplift curve only supported for a single treatment.")
        
    if pos_label is not None:
        y_true = (y_true == pos_label)

    y_score_c = y_score[trt==0]
    y_score_t = y_score[trt==1]
    y_true_c = y_true[trt==0]
    y_true_t = y_true[trt==1]
    
    idx_c, gains_c = _cumulative_gains_curve(y_true_c, y_score_c, sample_weight_c)
    idx_t, gains_t = _cumulative_gains_curve(y_true_t, y_score_t, sample_weight_t)

    # normalize
    n_c = (trt==0).sum()
    n_t = (trt==1).sum()
    if n_c == 0:
        raise RuntimeError("Cannot construct uplift curve: no cases in control")
    if n_t == 0:
        raise RuntimeError("Cannot construct uplift curve: no treated cases")
    gains_c /= n_c
    gains_t /= n_t

    # interpolate and subtract curves
    x_c = np.linspace(0, 1, n_c)
    if n_t == n_c:
        x = x_t = x_c
    else:
        x_t = np.linspace(0, 1, n_t)
        x = np.union1d(x_c, x_t)
    y_c = np.interp(x, idx_c/n_c, gains_c)
    y_t = np.interp(x, idx_t/n_t, gains_t)
    u = y_t - y_c
    return x, u

def uplift_curve_j(y_true, y_score, trt, n_trt=None, pos_label=None, sample_weight=None):
    """Uplift curve.

    Unless specified explicitly, y_true is assumed to be 0-1, with 1
    the positive outcome.

    This function implements the variant where scores are sorted
    jointly, see Verbeke, Nyberg, Verhelst.

    """
    y_true = check_array(y_true, ensure_2d=False)
    y_score = check_array(y_score, ensure_2d=False)
    trt, n_trt = check_trt(trt, n_trt)
    if sample_weight is None:
        check_consistent_length(y_true, y_score, trt)
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = check_array(sample_weight, ensure_2d=False)
        check_consistent_length(y_true, y_score, trt, sample_weight)
    if n_trt > 1:
        raise ValueError("uplift curve only supported for a single treatment.")
        
    if pos_label is not None:
        y_true = (y_true == pos_label)

    # normalize weights
    n_c = (trt==0).sum()
    n_t = (trt==1).sum()
    if n_c == 0:
        raise RuntimeError("Cannot construct uplift curve: no cases in control")
    if n_t == 0:
        raise RuntimeError("Cannot construct uplift curve: no treated cases")
    sample_weight[trt==0] /= -n_c
    sample_weight[trt==1] /= n_t
    
    idx, u = _cumulative_gains_curve(y_true, y_score, sample_weight)
    x = np.linspace(0, 1, len(idx))

    return x, u


def _auuc(uc_func, y_true, y_score, trt, n_trt=None, pos_label=None, sample_weight=None,
         subtract_diag=True):
    x, u = uc_func(y_true, y_score, trt, n_trt=n_trt, pos_label=pos_label,
                   sample_weight=sample_weight)
    auuc = np.trapz(u, x)
    if subtract_diag:
        a = x[-1] - x[0]
        h = u[-1]
        auuc -= a*h/2
    return auuc
def auuc(y_true, y_score, trt, n_trt=None, pos_label=None, sample_weight=None,
         subtract_diag=True):
    x, u = uplift_curve(y_true, y_score, trt, n_trt=n_trt, pos_label=pos_label,
                        sample_weight=sample_weight)
    return area_under_curve(x, u, subtract_diag=subtract_diag)
def auuc_j(y_true, y_score, trt, n_trt=None, pos_label=None, sample_weight=None,
           subtract_diag=True):
    x, u = uplift_curve_j(y_true, y_score, trt, n_trt=n_trt, pos_label=pos_label,
                          sample_weight=sample_weight)
    return area_under_curve(x, u, subtract_diag=subtract_diag)
