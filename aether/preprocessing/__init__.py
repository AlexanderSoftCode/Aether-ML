from aether._utils.lazy import install_lazy_attrs

_PREPROCESSING_MODULES = {
    "to_tensor":      "aether.preprocessing.transforms",
    "ToTensor":       "aether.preprocessing.transforms",
    "StandardScaler": "aether.preprocessing.transforms",
    "Compose":        "aether.preprocessing.transforms",
    "Rescale":        "aether.preprocessing.transforms",
    "Preprocess":     "aether.preprocessing.transforms",
    "deserialize":    "aether.preprocessing.transforms",
}

install_lazy_attrs(globals(), _PREPROCESSING_MODULES)
