from string import Template

import numpy as np

import aether.config as config

# --- Fused Adam/AdamW update kernel ----------------------------------------------

_ADAM_W_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const float bc1,
    const float bc2,
    const float weight_decay,
    const float l1_reg,
    const float l2_reg,
    const int N
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= N) return;

    float g = grad[out_idx];
    float p = param[out_idx];
    float m_val = m[out_idx];
    float v_val = v[out_idx];

    if (l1_reg > 0.0f) {
        g += l1_reg * (p < 0.0f ? -1.0f : 1.0f);
    }
    if (l2_reg > 0.0f) {
        g += l2_reg * p;
    }
    if (weight_decay > 0.0f) {
        p -= lr * weight_decay * p;
    }

    m_val = b1 * m_val + (1.0f - b1) * g;
    v_val = b2 * v_val + (1.0f - b2) * (g * g);

    float m_hat = m_val / bc1;
    float v_hat = v_val / bc2;

    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    param[out_idx] = p;
    m[out_idx] = m_val;
    v[out_idx] = v_val;
}   
''')

_ADAMW_CUDA = dict(hip_include="")
_ADAMW_HIP = dict(hip_include="#include <hip/hip_runtime.h>\n")

_adamw_kernel_cache = {}


def _get_compiled_adamw_kernel(variant: str):
    """Cached, compiled fused AdamW RawKernel for the given vendor variant.

    Single kernel signature 'variant' ('cuda' or 'hip') is the only cache axis,
    the same memoization pattern as pooling_kernel.py's
    _get_compiled_max_backward_kernel. A failed compile is cached as None so
    a broken build doesn't re-attempt (and re-warn) on every optimizer step.
    """
    if variant in _adamw_kernel_cache:
        return _adamw_kernel_cache[variant]

    vendor = _ADAMW_HIP if variant == "hip" else _ADAMW_CUDA
    kernel_name = f"fused_adamw_update_{variant}_kernel"
    source = _ADAM_W_TEMPLATE.substitute(
        kernel_name=kernel_name,
        **vendor,
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"fused_adamw_update_{variant}",
    )
    _adamw_kernel_cache[variant] = kernel
    return kernel


def is_gpu_adamw_available(variant: str) -> bool:
    """Checks if CuPy hardware support and the fused kernel are loaded/compiled."""
    return _get_compiled_adamw_kernel(variant) is not None


def launch_adamw_update(
    kernel,
    param, grad, momentum, cache,
    lr, beta1, beta2, eps,
    bias_correction1, bias_correction2,
    weight_decay, l1_reg, l2_reg,
    block_size: int = 512,
):
    """Launches the fused AdamW update kernel over a flat parameter buffer.

    Assumes `param`/`grad`/`momentum`/`cache` are C-contiguous float32 arrays
    of identical shape -- true by construction at every allocation site in
    this project (Dense/Conv2d weight init)
    """
    size = param.size
    blocks = (size + block_size - 1) // block_size

    kernel(
        (blocks,), (block_size,),
        (
            param, grad, momentum, cache,
            np.float32(lr), np.float32(beta1), np.float32(beta2), np.float32(eps),
            np.float32(bias_correction1), np.float32(bias_correction2),
            np.float32(weight_decay), np.float32(l1_reg), np.float32(l2_reg),
            np.int32(size),
        ),
    )