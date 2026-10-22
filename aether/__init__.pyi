# aether/__init__.pyi

# Static-typing companion to aether/__init__.py.
# Never executed -- exists purely so each public class
# gets real autocomplete/type info despite the runtime 
# __getattr__ lazy loading.

from aether.layers.activations import LeakyReLU as LeakyReLU
from aether.layers.activations import ReLU as ReLU
from aether.layers.activations import SoftMax as SoftMax
from aether.layers.conv import Conv2d as Conv2d
from aether.layers.dropout import Dropout as Dropout
from aether.layers.dropout import SpatialDropout as SpatialDropout
from aether.layers.linear import Dense as Dense
from aether.layers.linear import Flatten as Flatten
from aether.layers.normalization import BatchNorm as BatchNorm
from aether.layers.pooling import AvgPool2d as AvgPool2d
from aether.layers.pooling import GlobalAvgPool as GlobalAvgPool
from aether.layers.pooling import MaxPool2d as MaxPool2d
from aether.losses.categorical_crossentropy import (
    CategoricalCrossEntropy as CategoricalCrossEntropy,
)
from aether.losses.categorical_crossentropy import Loss as Loss
from aether.losses.categorical_crossentropy import (
    SoftmaxCategoricalCrossEntropy as SoftmaxCategoricalCrossEntropy,
)

# Direct module imports resolve stub lookups instantly:
from aether.metrics.accuracy import (
    CategoricalAccuracy as CategoricalAccuracy,
)
from aether.metrics.accuracy import (
    RegressionAccuracy as RegressionAccuracy,
)
from aether.model import Model as Model
from aether.optimizers.adam import Adam as Adam
from aether.optimizers.adam import AdamW as AdamW
from aether.preprocessing.transforms import Compose as Compose
from aether.preprocessing.transforms import Rescale as Rescale
from aether.preprocessing.transforms import StandardScaler as StandardScaler
from aether.preprocessing.transforms import ToTensor as ToTensor
from aether.preprocessing.transforms import to_tensor as to_tensor

__all__ = [
    # model
    "Model",
    # layers
    "Dense",
    "Flatten",
    "Conv2d",
    "ReLU",
    "LeakyReLU",
    "SoftMax",
    "MaxPool2d",
    "AvgPool2d",
    "GlobalAvgPool",
    "Dropout",
    "SpatialDropout",
    "BatchNorm",
    # losses
    "Loss",
    "CategoricalCrossEntropy",
    "SoftmaxCategoricalCrossEntropy",
    # metrics
    "CategoricalAccuracy",
    "RegressionAccuracy",
    # optimizers
    "Adam",
    "AdamW",
    # preprocessing
    "to_tensor",
    "ToTensor",
    "StandardScaler",
    "Rescale",
    "Compose"
]