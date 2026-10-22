from string import Template

import aether.config as config

# thread_idx = local coordinate inside the block
# blockDim = size/capacity inside a single block (512 or 1024) runtime 
# blockIdx = global block coordinate inside the grid
_POOL_FORWARD_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    const float* __restrict__ x,
    float* __restrict__ out,
    $aux_out_decl
    const int S, const int H_pad, const int W_pad, const int C,
    const int fH, const int fW, const int sH, const int sW,
    const int H_out, const int W_out,
    const unsigned int magic_scale, const int magic_shift  
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= C || w_out >= W_out) return;

    int h_s = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned long long prod = (unsigned long long)h_s * magic_scale;
    int s = (int)(prod >> (32 + magic_shift));
    if (s >= S) return;

    int h_out = h_s - (s * H_out);

    int h_start = h_out * sH;
    int w_start = w_out * sW;
    int batch_offset = s * H_pad * W_pad * C;

    $reduce_init

    for (int fh = 0; fh < fH; ++fh) {
        int h_in = h_start + fh;
        int row_offset = batch_offset + (h_in * W_pad) * C;

        for (int fw = 0; fw < fW; ++fw) {
            int w_in = w_start + fw;
            int in_idx = row_offset + w_in * C + c;

            $reduce_body
        }
    }

    int out_idx = ((s * H_out + h_out) * W_out + w_out) * C + c;
    $writeback
}
''')

_MAX_BACKWARD_NONOVERLAP_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    const float* __restrict__ dvalues,
    const int* __restrict__ max_indices,
    float* __restrict__ dinputs,
    const int S, const int H_pad, const int W_pad, const int C,
    const int fH, const int fW, const int sH, const int sW,
    const int H_out, const int W_out,
    const unsigned int magic_scale, const int magic_shift  
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= C || w_out >= W_out) return;

    int h_s = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned long long prod = (unsigned long long)h_s * magic_scale;
    int s = (int)(prod >> (32 + magic_shift));
    if (s >= S) return;

    int h_out = h_s - (s * H_out);

    int out_idx = ((s * H_out + h_out) * W_out + w_out) * C + c;
    int src_idx = max_indices[out_idx];
    dinputs[src_idx] = dvalues[out_idx];
}
''')

_AVG_BACKWARD_NONOVERLAP_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    const float* __restrict__ dvalues,
    float* __restrict__ dinputs,
    const int S, const int H_pad, const int W_pad, const int C,
    const int fH, const int fW, const int sH, const int sW,
    const int H_out, const int W_out,
    const unsigned int magic_scale, const int magic_shift  
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= C || w_out >= W_out) return;

    int h_s = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned long long prod = (unsigned long long)h_s * magic_scale;
    int s = (int)(prod >> (32 + magic_shift));
    if (s >= S) return;

    int h_out = h_s - (s * H_out);

    int h_start = h_out * sH;
    int w_start = w_out * sW;
    int batch_offset = s * H_pad * W_pad * C;

    int out_idx = ((s * H_out + h_out) * W_out + w_out) * C + c;
    float avg_val = dvalues[out_idx] / (float)(fH * fW);

    for (int fh = 0; fh < fH; ++fh) {
        int h_in = h_start + fh;
        int row_offset = batch_offset + (h_in * W_pad) * C;

        for (int fw = 0; fw < fW; ++fw) {
            int w_in = w_start + fw;
            int in_idx = row_offset + w_in * C + c;
            dinputs[in_idx] = avg_val;
        }
    }
}
''')

_GAP_FORWARD_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int S, const int H, const int W, const int C,
    float inv_area
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; 
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= C || s >= S) return;

    int out_idx = s * C + c;
    int batch_offset = s * (H * W * C);
    float sum = 0.0f; 
    for(int h = 0; h < H; ++h){
        int row_offset = batch_offset + h * (W * C);
        for(int w = 0; w < W; ++w){ 
            int flat_idx = row_offset + w * C + c;
            sum += x[flat_idx];
        }
    }

    out[out_idx] = sum * inv_area;
}
''')

_GAP_BACKWARD_TEMPLATE = Template(r'''
$hip_include
extern "C" __global__
void $kernel_name(
    const float* __restrict__ dvalues,
    float* __restrict__ dinputs,
    const int S, const int H, const int W, const int C,
    float inv_area
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y; 
    
    if (c >= C || s >= S) return; 
    int dout_idx = s * C + c;
    float scaled_grad = dvalues[dout_idx] * inv_area;
    int batch_offset = s * (H * W * C);

    for(int h = 0; h < H; ++h){
        int row_offset = batch_offset + h * (W * C);
        for(int w = 0; w < W; ++w){
            int flat_idx = row_offset + w * C + c;
            dinputs[flat_idx] = scaled_grad;
        }
    }
}

''')
_MAX_OP = {
    "name": "max",
    "aux_out_decl": "int* __restrict__ max_indices,",
    "reduce_init": (
        "int initial_idx = batch_offset + (h_start * W_pad + w_start) * C + c;\n"
        "    float max_val = x[initial_idx];\n"
        "    int best_idx = initial_idx;"
    ),
    "reduce_body": (
        "float val = x[in_idx];\n"
        "            if (val > max_val) { max_val = val; best_idx = in_idx; }"
    ),
    "writeback": "out[out_idx] = max_val;\n    max_indices[out_idx] = best_idx;",
}

_MAX_OP_INFERENCE = {
    "name": "max_inference",
    "aux_out_decl": "",
    "reduce_init": (
        "int initial_idx = batch_offset + (h_start * W_pad + w_start) * C + c;\n"
        "    float max_val = x[initial_idx];"
    ),
    "reduce_body": (
        "float val = x[in_idx];\n"
        "            if (val > max_val) { max_val = val; }"
    ),
    "writeback": "out[out_idx] = max_val;",
}

_AVG_OP = {
    "name": "avg",
    "aux_out_decl": "",
    "reduce_init": "float sum_val = 0.0f;",
    "reduce_body": "sum_val += x[in_idx];",
    "writeback": "out[out_idx] = sum_val / (float)(fH * fW);",
}


_pool_kernel_cache = {}

def _get_compiled_forward_kernel(op_dict: dict, variant: str):
    """Retrieves or compiles a GPU pooling RawKernel for the specified backend.

    Utilizes Just-In-Time (JIT) string metaprogramming and memoization. On first
    invocation for a given operation and variant, it dynamically injects 
    operation-specific reduction logic and target platform headers into 
    `_POOL_FORWARD_TEMPLATE`, compiles the raw C++ code via CuPy/NVRTC/HIPRTC, 
    and caches the resulting executable object. Subsequent calls fetch the 
    pre-compiled kernel directly from `_pool_kernel_cache`.

    Args:
        op_dict (dict): Descriptor containing operation-specific C++ code fragments.
            Expected keys:
                - "name" (str): Operation identifier (e.g., 'max', 'avg').
                - "aux_out_decl" (str): C++ declaration for auxiliary output 
                  buffers (e.g., max index pointers).
                - "reduce_init" (str): C++ setup code prior to the window loop.
                - "reduce_body" (str): C++ logic executed for each window element.
                - "writeback" (str): C++ code to commit final values to global memory.
        variant (str): Target GPU platform architecture. Must be either 'cuda' or 'hip'.

    Returns:
        RawKernel: A compiled CuPy RawKernel instance ready to be launched 
        with grid and block dimensions.
    """

    cache_key = (op_dict["name"], variant)
    if cache_key in _pool_kernel_cache:
        return _pool_kernel_cache[cache_key]

    kernel_name = f"pool2d_forward_{op_dict['name']}_{variant}_kernel"
    source = _POOL_FORWARD_TEMPLATE.substitute(
        hip_include="#include <hip/hip_runtime.h>\n" if variant == "hip" else "",
        kernel_name=kernel_name,
        aux_out_decl=op_dict["aux_out_decl"],
        reduce_init=op_dict["reduce_init"],
        reduce_body=op_dict["reduce_body"],
        writeback=op_dict["writeback"],
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"pool2d_forward_{op_dict['name']}_{variant}",
    )
    _pool_kernel_cache[cache_key] = kernel
    return kernel

def _get_compiled_max_backward_kernel(variant: str):
    """Compiles/retrieves the MaxPool2d non-overlapping backward RawKernel.

    Unlike forward compilation, this does not go through op-dict substitution —
    _MAX_BACKWARD_NONOVERLAP_TEMPLATE is a fully specialized kernel (direct
    argmax-indexed write, no per-window loop needed since max_indices already
    holds the flat source index from the forward pass). Only $hip_include and
    $kernel_name are templated.
    """
    cache_key = ("max_backward_nonoverlap", variant)
    if cache_key in _pool_kernel_cache:
        return _pool_kernel_cache[cache_key]

    kernel_name = f"pool2d_backward_max_nonoverlap_{variant}_kernel"
    source = _MAX_BACKWARD_NONOVERLAP_TEMPLATE.substitute(
        hip_include="#include <hip/hip_runtime.h>\n" if variant == "hip" else "",
        kernel_name=kernel_name,
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"pool2d_backward_max_nonoverlap_{variant}",
    )
    _pool_kernel_cache[cache_key] = kernel
    return kernel


def _get_compiled_avg_backward_kernel(variant: str):
    """Compiles/retrieves the AvgPool2d non-overlapping backward RawKernel.

    Same story as the max variant: _AVG_BACKWARD_NONOVERLAP_TEMPLATE is fully
    specialized (uniform fH*fW scatter, no argmax bookkeeping), only
    $hip_include and $kernel_name are templated.
    """
    cache_key = ("avg_backward_nonoverlap", variant)
    if cache_key in _pool_kernel_cache:
        return _pool_kernel_cache[cache_key]

    kernel_name = f"pool2d_backward_avg_nonoverlap_{variant}_kernel"
    source = _AVG_BACKWARD_NONOVERLAP_TEMPLATE.substitute(
        hip_include="#include <hip/hip_runtime.h>\n" if variant == "hip" else "",
        kernel_name=kernel_name,
    )

    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"pool2d_backward_avg_nonoverlap_{variant}",
    )
    _pool_kernel_cache[cache_key] = kernel
    return kernel

# Helper Functions for MaxPool2D
def get_max_pool2d_forward_kernel(variant: str, training: bool = True):
    """Returns a compiled RawKernel for the given variant ('cuda' or 'hip').
    Memoized via _pool_kernel_cache — no lazy self-rewriting needed.
    If we're not training, remove the 4D max_indicies tensor using a similar
    kernel _MAX_OP_INFERENCE
    """
    op = _MAX_OP if training else _MAX_OP_INFERENCE
    return _get_compiled_forward_kernel(op, variant)

def get_max_pool2d_backward_kernel(variant: str):
    """Returns a compiled RawKernel implementing the MaxPool2d backward
    scatter for the non-overlapping-window case (filter_size == stride
    only). Every padded-input position is the argmax target of at most
    one output window in this regime, so the kernel does a direct write
    instead of an atomicAdd. Do NOT use this for overlapping windows —
    that requires the atomic/scatter-add fallback path instead.
    """
    return _get_compiled_max_backward_kernel(variant)

# Helper Functions for AvgPool2d
def get_avg_pool2d_forward_kernel(variant: str):
    """Returns a compiled RawKernel for the given variant ('cuda' or 'hip').
    Memoized via _pool_kernel_cache — no lazy self-rewriting needed.
    """
    return _get_compiled_forward_kernel(_AVG_OP, variant)

def get_avg_pool2d_backward_kernel(variant: str):
    """Returns a compiled RawKernel implementing the AvgPool2d backward
    scatter for the non-overlapping-window case (filter_size == stride
    only). Do NOT use this for overlapping windows —
    that requires the atomic/scatter-add fallback path instead.
    """
    return _get_compiled_avg_backward_kernel(variant)

# Helper Functions for GlobalAvgPool

def get_gap_forward_kernel(variant: str):
    """Returns a compile RawKernel/Reduction kernel for GlobalAvgPool forward
    pass"""

    return _get_compiled_gap_forward_kernel(variant)

def get_gap_backward_kernel(variant: str):
    """Returns a compiled RawKernel for GlobalAvgPool backward"""

    return _get_compiled_gap_backward_kernel(variant)
def _get_compiled_gap_forward_kernel(variant: str):
    """Compiles/retrieves the GlobalAvgPool, memoizing it inside a dictionary lookup
    """
    cache_key = ("gap_forward", variant)
    if cache_key in _pool_kernel_cache:
        return _pool_kernel_cache[cache_key]

    kernel_name = f"gap_forward_{variant}_kernel"
    source = _GAP_FORWARD_TEMPLATE.substitute(
        hip_include="#include <hip/hip_runtime.h>\n" if variant == "hip" else "",
        kernel_name = kernel_name,
    )
    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"gap_forward_{kernel_name}"
    )

    _pool_kernel_cache[cache_key] = kernel
    return kernel 

def _get_compiled_gap_backward_kernel(variant: str): 

    """Compiles/retrieves the GlobalAvgPool, memoizing it inside a dictionary lookup
    """
    cache_key = ("gap_backward", variant)
    if cache_key in _pool_kernel_cache:
        return _pool_kernel_cache[cache_key]

    kernel_name = f"gap_backward_{variant}_kernel"
    source = _GAP_BACKWARD_TEMPLATE.substitute(
        hip_include="#include <hip/hip_runtime.h>\n" if variant == "hip" else "",
        kernel_name = kernel_name,
    )
    kernel = config.build_kernel(
        lambda: config.cp.RawKernel(source, kernel_name),
        name=f"gap_backward_{kernel_name}"
    )

    _pool_kernel_cache[cache_key] = kernel
    return kernel 
