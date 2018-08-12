"""Metrics to assess performance of uplift regression task

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

import numpy as np

from sklearn.utils.validation import check_array, check_consistent_length

def e_sate(y_true, y_pred, trt, n_trt=1):
    """Absolute error on Sample Average Treatment Effect.

    For multiple treatments, return weighted average."""
    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    check_consistent_length(y_true, y_pred, trt)

    average_ys_true = []
    sate_pred_s = []
    nts = []
    for t in range(n_trt + 1):
        ind = (trt==t)
        nt = ind.sum()
        if nt > 0:
            av_y_t_true = np.average(y_true[ind])
            if t > 0:
                sate_pred = np.average(y_pred[ind][:,t-1]) # ATT
                #sate_pred = np.average(y_pred[:,t-1]) # ATE
            else:
                sate_pred = np.nan
        else:
            av_y_t_true = np.nan
            av_y_t_pred = np.nan
        nts.append(nt)
        average_ys_true.append(av_y_t_true)
        sate_pred_s.append(sate_pred)
    if nts[0] == 0:
        raise RuntimeError("Cannot estimate ATE. No cases in control")
    n_treated = sum(nts[1:])
    if n_treated == 0:
        raise RuntimeError("Cannot estimate ATE. No cases in any treatment")
    average_y_control = average_ys_true[0]
    sate_pred_s = sate_pred_s[1:]
    sate_true_s = [average_ys_true[t+1] - average_y_control
                       for t in range(n_trt)]
    return sum(nt/n_treated * abs(sate_pred - sate_true)
                   for nt, sate_pred, sate_true in
                   zip(nts[1:], sate_pred_s, sate_true_s))
