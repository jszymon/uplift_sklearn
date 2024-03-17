"""Measures based on comparing treatment and control statistics within
bins, such as quantiles."""

import numpy as np

def iter_quantiles(scores, n=10, joint=False):
    """Iterate simultaneously over quantiles of several vectors of
    scores.

    Returns a generator which, for each quantile, returns a list of
    index arrays for each score vector.

    If joint is True, quantiles are computed jointly on all
    concatenated scores.

    """
    counts = []
    idxs = []
    if joint:
        q = np.quantile(np.concatenate(scores),
                        np.linspace(0,1,n, endpoint=False))
    for s in scores:
        if not joint:
            q = np.quantile(s, np.linspace(0,1,n, endpoint=False))
        b = np.digitize(s, q)-1
        c = np.r_[[0], np.cumsum(np.bincount(b, minlength=n))]
        idx = np.argsort(b)
        counts.append(c)
        idxs.append(idx)
    for i in range(n):
        yield [idx[c[i]:c[i+1]] for idx, c in zip(idxs, counts)]

#def QMSE(y_true, y_score, trt, n_trt=None, sample_weight=None,
#         n_bins=10, allow_nans=False, joint_quantiles=False):
#    #scores = 
#    for q_idxs in iter_quantiles([range(10), range(5)], 3, joint=joint_quantiles)
