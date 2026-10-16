from aether._utils.lazy import install_lazy_attrs

_OPTIMIZER_MODULES = {
    "Optimizer": "aether.optimizers.adam",
    "Adam":  "aether.optimizers.adam",
    "AdamW": "aether.optimizers.adam",
}

install_lazy_attrs(globals(), _OPTIMIZER_MODULES)
