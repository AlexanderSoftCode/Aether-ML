from importlib.metadata import PackageNotFoundError, version as _dist_version

from aether._utils._lazy import install_lazy_attrs
from aether.layers import _LAYER_MODULES
from aether.losses import _LOSS_MODULES
from aether.metrics import _ACC_MODULES
from aether.optimizers import _OPTIMIZER_MODULES
from aether.preprocessing import _PREPROCESSING_MODULES

try:
    __version__ = _dist_version("aether-ml")
except PackageNotFoundError:
    # Running from a source checkout that was never pip-installed. Keep the
    # attribute defined so `ae.__version__` never falls through to the lazy
    # __getattr__ and raises a misleading AttributeError.
    __version__ = "0.0.0.dev0"

_MODEL_MODULES = {
    "Model": "aether.model",
}
_TOP_LEVEL_MODULES = {
    **_MODEL_MODULES,
    **_LAYER_MODULES,
    **_LOSS_MODULES,
    **_ACC_MODULES,
    **_OPTIMIZER_MODULES,
    **_PREPROCESSING_MODULES
}
 
install_lazy_attrs(globals(), _TOP_LEVEL_MODULES) 

# Clean up private setup variables so ae.<tab> remains clean
del (
    install_lazy_attrs,
    _dist_version,
    PackageNotFoundError,
    _LAYER_MODULES, 
    _LOSS_MODULES, 
    _ACC_MODULES,
    _OPTIMIZER_MODULES, 
    _MODEL_MODULES, 
    _PREPROCESSING_MODULES,
    _TOP_LEVEL_MODULES,
)