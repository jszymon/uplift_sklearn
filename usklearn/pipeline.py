import sklearn.pipeline

class Pipeline(sklearn.pipeline.Pipeline):
    def fit(self, X, y=None, trt=None, n_trt=None, **fit_params):
        for n in self.named_steps:
            fit_params[n + "__trt"] = trt
            fit_params[n + "__n_trt"] = n_trt
        super().fit(X, y, **fit_params)
