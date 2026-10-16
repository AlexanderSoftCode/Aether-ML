from aether._utils.lazy import install_lazy_attrs

_ACC_MODULES = {
    "Accuracy":                "aether.metrics.accuracy",
    "CategoricalAccuracy":     "aether.metrics.accuracy",
    "RegressionAccuracy":      "aether.metrics.accuracy"
}

install_lazy_attrs(globals(), _ACC_MODULES)
