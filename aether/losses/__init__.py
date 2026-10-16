from aether._utils.lazy import install_lazy_attrs

_LOSS_MODULES = {
    "Loss":                                         "aether.losses.categorical_crossentropy",
    "CategoricalCrossEntropy":                      "aether.losses.categorical_crossentropy",
    "SoftmaxCategoricalCrossEntropy":               "aether.losses.categorical_crossentropy"
}
install_lazy_attrs(globals(), _LOSS_MODULES)
