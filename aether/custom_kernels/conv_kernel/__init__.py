"""Matrix-core (WMMA) convolution kernels, one module per kernel.

common   -- shape metadata, launch plans, tile constants, vendor dicts
forward  -- conv_forward_wmma
dweight  -- conv_backward_dweight_wmma and its split-K sizing
dinput   -- conv_backward_dinput_wmma
"""
from aether.custom_kernels.conv_kernel.common import (
    ConvLaunchPlan,
    ConvShapeMeta,
    get_is_conv_gpu_available,
)
from aether.custom_kernels.conv_kernel.dinput import (
    _CONV_BACKWARD_DINPUT_CACHE,
    get_compiled_backward_dinput_kernel,
    plan_dinput_launch,
)
from aether.custom_kernels.conv_kernel.dweight import (
    _CONV_BACKWARD_DWEIGHT_CACHE,
    choose_dweight_split_k,
    get_compiled_backward_dweight_kernel,
    plan_dweight_launch,
)
from aether.custom_kernels.conv_kernel.forward import (
    _CONV_FORWARD_KERNEL_CACHE,
    get_compiled_forward_conv_kernel,
    plan_forward_launch,
)

__all__ = [
    "_CONV_BACKWARD_DINPUT_CACHE",
    "_CONV_BACKWARD_DWEIGHT_CACHE",
    "_CONV_FORWARD_KERNEL_CACHE",
    "ConvLaunchPlan",
    "ConvShapeMeta",
    "choose_dweight_split_k",
    "get_compiled_backward_dinput_kernel",
    "get_compiled_backward_dweight_kernel",
    "get_compiled_forward_conv_kernel",
    "get_is_conv_gpu_available",
    "plan_dinput_launch",
    "plan_dweight_launch",
    "plan_forward_launch",
]
