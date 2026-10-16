from aether._utils.lazy import install_lazy_attrs

_LAYER_MODULES = {
    "Dense":         "aether.layers.linear",
    "Flatten":       "aether.layers.linear",
    "Conv2d":          "aether.layers.conv",
    "ReLU":          "aether.layers.activations",
    "LeakyReLU":     "aether.layers.activations",
    "SoftMax":       "aether.layers.activations",
    "MaxPool2d":      "aether.layers.pooling",
    "AvgPool2d":      "aether.layers.pooling",
    "GlobalAvgPool":  "aether.layers.pooling",
    "Dropout":        "aether.layers.dropout",
    "SpatialDropout": "aether.layers.dropout",
    "BatchNorm":      "aether.layers.normalization",
}

install_lazy_attrs(globals(), _LAYER_MODULES)
