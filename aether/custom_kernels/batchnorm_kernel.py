from string import Template

import aether.config as config

BLOCK_Y = 8


# Shared by every kernel below. block_sum adds each thread's (a, b) pair over the
# block's BLOCK_Y rows and returns true on the row-0 threads that hold a channel's
# total; sum_partials does the same over the G rows of a partials[2, G, C] buffer.
_PRELUDE = r'''
$hip_include
#define WAVEFRONT 32
#define BLOCK_Y   $block_y

__device__ __forceinline__ bool block_sum(float& a, float& b, int c, int C) {
    __shared__ float sh_a[BLOCK_Y][WAVEFRONT];
    __shared__ float sh_b[BLOCK_Y][WAVEFRONT];

    sh_a[threadIdx.y][threadIdx.x] = a;
    sh_b[threadIdx.y][threadIdx.x] = b;
    __syncthreads();

    if (threadIdx.y != 0 || c >= C) return false;
    a = 0.0f;
    b = 0.0f;
    for (int j = 0; j < BLOCK_Y; ++j) {
        a += sh_a[j][threadIdx.x];
        b += sh_b[j][threadIdx.x];
    }
    return true;
}

__device__ __forceinline__ bool sum_partials(
    const float* __restrict__ partials, int G, int c, int C, float& a, float& b
) {
    a = 0.0f;
    b = 0.0f;
    if (c < C) {
        for (int g = threadIdx.y; g < G; g += BLOCK_Y) {
            a += partials[g * C + c];
            b += partials[(G + g) * C + c];
        }
    }
    return block_sum(a, b, c, C);
}
'''

# Each block reduces its rows of [N, C] and writes one (a, b) pair per channel to
# partials[2, gridDim.y, C]. No atomics, so the result is deterministic.
_STATS_REDUCE_TEMPLATE = Template(_PRELUDE + r'''
extern "C" __global__
void $kernel_name(
    const float* __restrict__ p,
    const float* __restrict__ q,      // unused (nullptr) ops still take the param
    const float* __restrict__ mean,   // channel mean [C], unused ops pass nullptr
    float* __restrict__ partials,     // [2, gridDim.y, C]
    const int N,
    const int C
) {
    int c = blockIdx.x * WAVEFRONT + threadIdx.x;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    if (c < C) {
        $prologue
        for (int n = blockIdx.y * BLOCK_Y + threadIdx.y; n < N; n += gridDim.y * BLOCK_Y) {
            int idx = n * C + c;
            $accumulate
        }
    }

    if (!block_sum(acc_a, acc_b, c, C)) return;
    partials[blockIdx.y * C + c] = acc_a;
    partials[(gridDim.y + blockIdx.y) * C + c] = acc_b;
}
''')

# Reduces the stats partials and finishes the per-channel math in one launch:
# stats[4, C] = (mean, var, inv_std, scale = gamma * inv_std), and the running
# statistics are updated in place.
_FORWARD_FINALIZE_TEMPLATE = Template(_PRELUDE + r'''
extern "C" __global__
void $kernel_name(
    const float* __restrict__ partials,   // [2, G, C] sums of d = x - x[0, c] and d * d
    const float* __restrict__ x,          // [N, C], row 0 is the shift
    const float* __restrict__ gamma,
    float* __restrict__ stats,            // [4, C]
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    const int G,
    const int N,
    const int C,
    const float eps,
    const float momentum
) {
    int c = blockIdx.x * WAVEFRONT + threadIdx.x;
    float s1, s2;
    if (!sum_partials(partials, G, c, C, s1, s2)) return;

    float d_mean = s1 / N;
    float mean = x[c] + d_mean;
    float var = fmaxf(s2 / N - d_mean * d_mean, 0.0f);
    float inv_std = 1.0f / sqrtf(var + eps);

    stats[c] = mean;
    stats[C + c] = var;
    stats[2 * C + c] = inv_std;
    stats[3 * C + c] = gamma[c] * inv_std;

    running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
    running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;
}
''')

# Reduces the grad_sums partials into grads[4, C] = (dgamma, dbeta, k1, k2), so
# that dx = dy * scale + (x - mean) * k1 + k2.
_BACKWARD_FINALIZE_TEMPLATE = Template(_PRELUDE + r'''
extern "C" __global__
void $kernel_name(
    const float* __restrict__ partials,   // [2, G, C] sums of dy and dy * (x - mean)
    const float* __restrict__ inv_std,
    const float* __restrict__ scale,
    float* __restrict__ grads,            // [4, C]
    const int G,
    const int N,
    const int C
) {
    int c = blockIdx.x * WAVEFRONT + threadIdx.x;
    float s1, s2;
    if (!sum_partials(partials, G, c, C, s1, s2)) return;

    float is = inv_std[c];
    float sc = scale[c];

    grads[c] = is * s2;
    grads[C + c] = s1;
    grads[2 * C + c] = -sc * is * is * s2 / N;
    grads[3 * C + c] = -sc * s1 / N;
}
''')

_BN_CUDA = dict(hip_include="")
_BN_HIP = dict(hip_include="#include <hip/hip_runtime.h>\n")

# Shifting by a sample of the channel keeps the one-pass variance from cancelling
# when |mean| >> std, without the per-element divide Welford needs.
_STATS_OP = {
    "name": "stats",
    "template": _STATS_REDUCE_TEMPLATE,
    "prologue": "float k = p[c];",
    "accumulate": "float d = p[idx] - k; acc_a += d; acc_b += d * d;",
}

_GRAD_SUMS_OP = {
    "name": "grad_sums",
    "template": _STATS_REDUCE_TEMPLATE,
    "prologue": "float mu = mean[c];",
    "accumulate": "float g = p[idx]; acc_a += g; acc_b += g * (q[idx] - mu);",
}

_FORWARD_FINALIZE_OP = {
    "name": "forward_finalize",
    "template": _FORWARD_FINALIZE_TEMPLATE,
}

_BACKWARD_FINALIZE_OP = {
    "name": "backward_finalize",
    "template": _BACKWARD_FINALIZE_TEMPLATE,
}

_bn_kernel_cache = {}


def _get_compiled_kernel(op_dict: dict, variant: str, block_y: int = BLOCK_Y):
    """Cached, compiled RawKernel for the given vendor variant.

    Memoized by (op_name, variant) without baking shapes into the compile key.
    """
    key = (op_dict["name"], variant)
    if key in _bn_kernel_cache:
        return _bn_kernel_cache[key]

    vendor = _BN_HIP if variant == "hip" else _BN_CUDA
    op_name = op_dict["name"]
    kernel_name = f"bn_{op_name}_{variant}_kernel"
    source = op_dict["template"].substitute(
        kernel_name=kernel_name,
        block_y=block_y,
        prologue=op_dict.get("prologue", ""),
        accumulate=op_dict.get("accumulate", ""),
        **vendor,
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"bn_{op_name}_{variant}",
    )
    _bn_kernel_cache[key] = kernel
    return kernel


def get_bn_stats_kernel(variant: str):
    """Returns a compiled RawKernel for the BatchNorm FP32 shifted sum / sum-of-squares partials."""
    return _get_compiled_kernel(_STATS_OP, variant)


def get_bn_forward_finalize_kernel(variant: str):
    """Returns a compiled RawKernel that turns the stats partials into mean, var, inv_std and scale."""
    return _get_compiled_kernel(_FORWARD_FINALIZE_OP, variant)


def get_bn_grad_sums_kernel(variant: str):
    """Returns a compiled RawKernel for BatchNorm FP32 gradient sums partials."""
    return _get_compiled_kernel(_GRAD_SUMS_OP, variant)


def get_bn_backward_finalize_kernel(variant: str):
    """Returns a compiled RawKernel that turns the gradient partials into dgamma, dbeta and the dx coefficients."""
    return _get_compiled_kernel(_BACKWARD_FINALIZE_OP, variant)

# --- Elementwise Kernels --------------------------------------------------------

# Training and inference share it: the centered form keeps the rounding error of
# folding (mean, scale, beta) into one bias out of the output.
_bn_apply_ew = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'float32 x, float32 mean, float32 scale, float32 beta',
        'float32 out',
        'out = fmaf(x - mean, scale, beta);',
        'bn_apply_ew',
    ),
    name='_bn_apply_ew',
)

_bn_backward_ew = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'float32 dvalues, float32 x, float32 mean, float32 scale, float32 k1, float32 k2',
        'float32 dinputs',
        'dinputs = dvalues * scale + (x - mean) * k1 + k2;',
        'bn_backward_ew',
    ),
    name='_bn_backward_ew',
)


# --- Availability Helper --------------------------------------------------------

def is_gpu_bn_available(variant: str = None) -> bool:
    """Checks if CuPy hardware support and BatchNorm kernels are available."""
    if variant is None:
        variant, _ = config.resolve_gpu_launch_geometry()

    return (
        get_bn_stats_kernel(variant) is not None
        and get_bn_forward_finalize_kernel(variant) is not None
        and get_bn_grad_sums_kernel(variant) is not None
        and get_bn_backward_finalize_kernel(variant) is not None
        and _bn_apply_ew is not None
        and _bn_backward_ew is not None
    )
