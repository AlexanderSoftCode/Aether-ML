import warnings

import aether.config as config
from aether.custom_kernels.conv_kernel.common import _COMPILE_OPTIONS

# One launch casts both fp16 operands of the conv backward kernels. dvalues is
# clipped to the fp16 range first; inputs is not, so dweight sees exactly the
# fp16 inputs the forward pass multiplied.
CONV_BACKWARD_CAST_SOURCE = r'''
$fp16_include

#define HALF_MAX 65504.0f

extern "C" __global__ void conv_backward_cast_fp16(
    const float* __restrict__ dvalues,     // [n_dvalues] fp32, contiguous
    half* __restrict__ dvalues_fp16,       // [n_dvalues]
    const long long n_dvalues,
    const float* __restrict__ inputs,      // [n_inputs] fp32, contiguous
    half* __restrict__ inputs_fp16,        // [n_inputs]
    const long long n_inputs
) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_dvalues) {
        const float v = dvalues[i];
        // Comparisons rather than fminf/fmaxf, which would turn NaN into
        // +-HALF_MAX: like xp.clip, NaN passes through and +-inf clamp.
        dvalues_fp16[i] = __float2half(v > HALF_MAX ? HALF_MAX : (v < -HALF_MAX ? -HALF_MAX : v));
    } else if (i < n_dvalues + n_inputs) {
        const long long j = i - n_dvalues;
        inputs_fp16[j] = __float2half(inputs[j]);
    }
}
'''

_CONV_BACKWARD_CAST_CACHE = {}


def get_compiled_backward_cast_kernel(is_hip: bool):
    if is_hip in _CONV_BACKWARD_CAST_CACHE:
        return _CONV_BACKWARD_CAST_CACHE[is_hip]

    fp16_include = "#include <hip/hip_fp16.h>" if is_hip else "#include <cuda_fp16.h>"
    source = CONV_BACKWARD_CAST_SOURCE.replace("$fp16_include", fp16_include)

    try:
        module = config.cp.RawModule(code=source, options=_COMPILE_OPTIONS)
        kernel = module.get_function('conv_backward_cast_fp16')
    except (config.cp.cuda.compiler.CompileException, config.cp.cuda.driver.CUDADriverError) as e:
        warnings.warn(f"[aether] conv_kernel: backward fp16 cast compile failed: {e}")
        _CONV_BACKWARD_CAST_CACHE[is_hip] = None
        return None

    _CONV_BACKWARD_CAST_CACHE[is_hip] = kernel
    return kernel


def cast_backward_operands_fp16(dvalues, inputs):
    """(dvalues clipped to +-65504, inputs) as fp16, in one launch.

    Both must be contiguous fp32. Returns None when the kernel is unavailable,
    so the caller can fall back to astype/clip.
    """
    kernel = get_compiled_backward_cast_kernel(config.IS_HIP)
    if kernel is None:
        return None

    xp = config.cp
    dvalues_fp16 = xp.empty(dvalues.shape, dtype=xp.float16)
    inputs_fp16 = xp.empty(inputs.shape, dtype=xp.float16)
    n_dvalues, n_inputs = dvalues.size, inputs.size

    _, threads = config.resolve_gpu_launch_geometry()
    blocks = -(-(n_dvalues + n_inputs) // threads)
    kernel(
        (blocks,), (threads,),
        (dvalues, dvalues_fp16, xp.int64(n_dvalues), inputs, inputs_fp16, xp.int64(n_inputs)),
    )
    return dvalues_fp16, inputs_fp16
