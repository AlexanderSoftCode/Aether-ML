from string import Template

import aether.config as config

BLOCK_Y = 8


_STATS_REDUCE_TEMPLATE = Template(r'''
$hip_include
#define WAVEFRONT 32
#define BLOCK_Y   $block_y

extern "C" __global__
void $kernel_name(
    const float* __restrict__ p,
    const float* __restrict__ q,      // unused (nullptr) ops still take the param
    const float* __restrict__ mean,   // channel mean [C], unused ops pass nullptr
    float* __restrict__ out_a,        // [C], caller must zero-init
    float* __restrict__ out_b,        // [C], caller must zero-init
    const int N,
    const int C
) {
    __shared__ float sh_a[BLOCK_Y][WAVEFRONT];
    __shared__ float sh_b[BLOCK_Y][WAVEFRONT];

    int c  = blockIdx.x * WAVEFRONT + threadIdx.x;
    int ty = threadIdx.y;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    if (c < C) {
        $mean_load
        for (int n = blockIdx.y * BLOCK_Y + ty; n < N; n += gridDim.y * BLOCK_Y) {
            int idx = n * C + c;
            float val_p = p[idx];
            $load_q
            acc_a += $accum_a_expr;
            $accum_b_stmt
        }
    }

    sh_a[ty][threadIdx.x] = acc_a;
    sh_b[ty][threadIdx.x] = acc_b;
    __syncthreads();

    if (ty == 0 && c < C) {
        float block_a = 0.0f;
        float block_b = 0.0f;
        for (int j = 0; j < BLOCK_Y; ++j) {
            block_a += sh_a[j][threadIdx.x];
            block_b += sh_b[j][threadIdx.x];
        }
        atomicAdd(&out_a[c], block_a);
        $writeback_b
    }
}
''')

_BN_CUDA = dict(hip_include="")
_BN_HIP = dict(hip_include="#include <hip/hip_runtime.h>\n")

_MEAN_OP = {
    "name": "mean",
    "mean_load": "",
    "load_q": "",
    "accum_a_expr": "val_p",
    "accum_b_stmt": "",
    "writeback_b": "",
}

_VAR_OP = {
    "name": "var",
    "mean_load": "float mu = mean[c];",
    "load_q": "",
    "accum_a_expr": "(val_p - mu) * (val_p - mu)",
    "accum_b_stmt": "",
    "writeback_b": "",
}

_GRAD_SUMS_OP = {
    "name": "grad_sums",
    "mean_load": "float mu = mean[c];",
    "load_q": "float val_q = q[idx];",
    "accum_a_expr": "val_p",
    "accum_b_stmt": "acc_b += val_p * (val_q - mu);",
    "writeback_b": "atomicAdd(&out_b[c], block_b);",
}

_bn_reduce_kernel_cache = {}


def _get_compiled_reduce_kernel(op_dict: dict, variant: str, block_y: int = BLOCK_Y):
    """Cached, compiled reduction RawKernel for the given vendor variant.

    Memoized by (op_name, variant) without baking shapes into the compile key.
    """
    key = (op_dict["name"], variant)
    if key in _bn_reduce_kernel_cache:
        return _bn_reduce_kernel_cache[key]

    vendor = _BN_HIP if variant == "hip" else _BN_CUDA
    op_name = op_dict["name"]
    kernel_name = f"bn_reduce_{op_name}_{variant}_kernel"
    source = _STATS_REDUCE_TEMPLATE.substitute(
        kernel_name=kernel_name,
        block_y=block_y,
        mean_load=op_dict["mean_load"],
        load_q=op_dict["load_q"],
        accum_a_expr=op_dict["accum_a_expr"],
        accum_b_stmt=op_dict["accum_b_stmt"],
        writeback_b=op_dict["writeback_b"],
        **vendor,
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"bn_reduce_{op_name}_{variant}",
    )
    _bn_reduce_kernel_cache[key] = kernel
    return kernel

def get_bn_mean_kernel(variant: str):
    """Returns a compiled RawKernel for BatchNorm FP32 mean reduction."""
    return _get_compiled_reduce_kernel(_MEAN_OP, variant)


def get_bn_var_kernel(variant: str):
    """Returns a compiled RawKernel for BatchNorm FP32 centered variance reduction."""
    return _get_compiled_reduce_kernel(_VAR_OP, variant)


def get_bn_grad_sums_kernel(variant: str):
    """Returns a compiled RawKernel for BatchNorm FP32 gradient sums reduction."""
    return _get_compiled_reduce_kernel(_GRAD_SUMS_OP, variant)

# --- Elementwise Finalize Kernels -----------------------------------------------

_bn_forward_ew = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'float32 x, float32 mean, float32 inv_std, float32 gamma, float32 beta',
        'float32 out',
        'out = (x - mean) * inv_std * gamma + beta;',
        'bn_forward_ew',
    ),
    name='_bn_forward_ew',
)

_bn_inference_ew = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'float32 x, float32 scale, float32 bias',
        'float32 out',
        'out = x * scale + bias;',
        'bn_inference_ew',
    ),
    name='_bn_inference_ew',
)

_bn_backward_ew = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'float32 dvalues, float32 x, float32 mean, float32 gamma_inv_std, float32 dvar_scaled, float32 dmu_over_n',
        'float32 dinputs',
        'dinputs = dvalues * gamma_inv_std + (x - mean) * dvar_scaled + dmu_over_n;',
        'bn_backward_ew',
    ),
    name='_bn_backward_ew',
)


# --- Availability Helper --------------------------------------------------------

def is_gpu_bn_available(variant: str = None) -> bool:
    """Checks if CuPy hardware support and BatchNorm kernels are available."""
    if variant is None:
        variant, _ = config.resolve_gpu_launch_geometry()

    mean_k = get_bn_mean_kernel(variant)
    var_k = get_bn_var_kernel(variant)
    grad_sums_k = get_bn_grad_sums_kernel(variant)
    return (
        mean_k is not None
        and var_k is not None
        and grad_sums_k is not None
        and _bn_forward_ew is not None
        and _bn_inference_ew is not None
        and _bn_backward_ew is not None
    )