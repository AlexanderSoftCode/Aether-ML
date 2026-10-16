"""Internal framework utilities. Never exposed outside aether"""

from aether._utils.lazy import install_lazy_attrs

_UTILS = {
    "install_lazy_attrs": "aether._utils.lazy",
    "NullAccuracy":       "aether._utils.null_objects",
    "NullOptimizer":      "aether._utils.null_objects",
    "NullPreprocessor":   "aether._utils.null_objects",
    "TrainingProgress":   "aether._utils.progress"
}

install_lazy_attrs(globals(), _UTILS)
