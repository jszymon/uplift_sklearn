"""Uplift models based on target transform."""

class TargetTransformUpliftRegressor(BaseEstimator, UpliftRegressorMixin):
    def __init__(self, base_estimator=LinearRegression()):
        self.base_estimator = base_estimator
    def fit(self, X, y, trt, n_trt=None):
        X, y = check_X_y(X, y, accept_sparse="csr")
        self.trt_, self.n_trt_ = check_trt(trt, n_trt)
        check_consistent_length(X, y, self.trt_)
        self.n_models_ = self.n_trt_ + 1
        self.models_ = []
        self.n_ = np.empty(self.n_models_, dtype=int)
        for i in range(self.n_models_):
            mi = clone(self.base_estimator)
            ind = (trt==i)
            self.n_[i] = ind.sum()
            Xi = X[ind]
            yi = y[ind]
            mi.fit(Xi, yi)
            self.models_.append(mi)
        return self
    def predict(self, X):
        y_control = self.models_[0].predict(X)
        cols = [self.models_[i+1].predict(X) - y_control
                    for i in range(self.n_trt_)]
        if self.n_trt_ == 1:
            y = cols[0]
        else:
            y = np.column_stack(cols)
        return y
    def predict_action(self, X):
        """Predict most beneficial action."""
        y = self.predict(X)
        if self.n_trt_ == 1:
            a = (y > 0)*1
        else:
            a = np.argmax(y, axis=1) + 1
            best_y = np.max(y, axis=1)
            a[best_y <= 0] == 0
        return a
