import numpy as np

from usklearn.metrics.bins import iter_quantiles

def test_iter_quantiles():
    trt =    np.array([0,0,0,0,0, 1,1,1,1,1])
    scores = np.array([1,2,3,4,5, 5,4,3,2,1], dtype=float)

    for i, q in enumerate(iter_quantiles(scores, trt, 1, n=5)):
        assert len(q) == 2
        assert q[0].shape == (1,)
        assert q[1].shape == (1,)
        assert q[0].item() == i
        assert q[1].item() == 9-i
