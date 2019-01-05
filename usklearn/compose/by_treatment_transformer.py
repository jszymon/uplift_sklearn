import numpy as np

from sklearn.base import clone

from ..base import UpliftTransformerMixin

# in the future treatment specific transforms will be implemented, similar to
# sklearn.compose.ColumnTransformer
class ByTreatmentTransformer(UpliftTransformerMixin):
    def __init__(self, remainder="passthrough")
        self.remainder=remainder
    def fit(self, X, trt, n_trt=None, y=None):
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.trt_, self.n_trt_ = check_trt(trt, n_trt)
        check_consistent_length(X, y, self.trt_)
        # validate remainder
        self.is_transformer_ = ((hasattr(self.remainder, "fit")
                                 or hasattr(self.remainder, "fit_transform"))
                                 and hasattr(self.remainder, "transform"))
        if (self.remainder not in ('drop', 'passthrough')
                and not is_transformer_):
            raise ValueError(
                "The remainder keyword needs to be one of 'drop', "
                "'passthrough', or estimator. '%s' was passed instead" %
                self.remainder)
        if self.is_transformer_:
            self.remainder_transformers = [None] * self.n_trt_
            for t in range(self.n_trt_ + 1):
                self.remainder_transformers[t] = clone(self.remainder)
                self.remainder_transformers[t].fit(X[trt==t], y[trt=t])
        return self
    def transform(self, X, trt, n_trt=None, y=None):
        trt_, n_trt_ = check_trt(trt, n_trt)
        if trt_.max() > self.n_trt_:
            raise ValueError("More treatment if transform than were fitted")
        if any(sparse.issparse(f) for f in Xs):
            raise NotImplemented() # TODO
        else:
            X_transf = np.emptylike(X)
            for t in range(self.n_trt_ + 1):
                tr = self.remainder_transformers[t]
                if y is None:
                    X_transf[trt_ == t] = tr.transform(X[trt==t])
                else:
                    X_transf[trt_ == t] = tr.transform(X[trt==t], y[trt=t])
        return X_transf
