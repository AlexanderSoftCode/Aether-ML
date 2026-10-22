# Static declaration for the IDEs
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

__all__ = [
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
]