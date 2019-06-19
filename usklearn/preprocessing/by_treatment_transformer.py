import numpy as np

from sklearn.base import clone
from sklearn.utils import check_X_y, check_consistent_length, check_array

from ..base import UpliftTransformerMixin
from ..utils import check_trt

class ByTreatmentTransformer(UpliftTransformerMixin):
    """Apply sklearn transformer to uplift data.

    Parameters
    ----------

    transformer : a sklearn transformer object or 'passthrough'.
    by_treatment : if True, apply transform within each treatment
        separately.
    """
    def __init__(self, transformer="passthrough", by_treatment=False):
        self.transformer = transformer
        self.by_treatment = by_treatment
    def fit(self, X, y=None, trt=None, n_trt=None):
        if y is not None:
            X, y = check_X_y(X, y)
        else:
            X = check_array(X)
        if self.transformer == "passthrough" or not self.by_treatment:
            check_consistent_length(X, y)
        else:
            trt, self.n_trt_ = check_trt(trt, n_trt)
            check_consistent_length(X, y, trt)
        # validate transform
        self.is_transformer_ = ((hasattr(self.transformer, "fit")
                                 or hasattr(self.transformer, "fit_transform"))
                                 and hasattr(self.transformer, "transform"))
        if (self.transformer != 'passthrough'
                and not self.is_transformer_):
            raise ValueError(
                "The transformer keyword needs to be either "
                "'passthrough', or estimator. '%s' was passed instead" %
                self.transform)
        if self.is_transformer_:
            if self.by_treatment:
                self.transformers_ = [None] * (self.n_trt_ + 1)
                for t in range(self.n_trt_ + 1):
                    self.transformers_[t] = clone(self.transformer)
                    yt = y[trt==t] if y is not None else None
                    self.transformers_[t].fit(X[trt==t], yt)
            else:
                self.transformers_ = [clone(self.transformer)]
                self.transformers_[0].fit(X, y)
        return self
    def transform(self, X, y=None, trt=None, n_trt=None):
        if self.is_transformer_ and self.by_treatment:
            trt_, n_trt_ = check_trt(trt, n_trt)
            if trt_.max() > self.n_trt_:
                raise ValueError("More treatments to transform"
                                 " than were fitted")
        if not self.is_transformer_:
            X_transf = X # passthrough
        elif not self.by_treatment:
            X_transf = self.transformers_[0].transform(X)
        else:
            X_transf = np.empty_like(X)
            for t in range(self.n_trt_ + 1):
                tr = self.transformers_[t]
                if y is None:
                    X_transf[trt_ == t] = tr.transform(X[trt_==t])
                else:
                    X_transf[trt_ == t] = tr.transform(X[trt_==t], y[trt_==t])
        return X_transf
