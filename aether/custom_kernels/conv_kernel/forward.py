import os
import warnings
from string import Template

import aether.config as config
from aether.custom_kernels.conv_kernel.common import (
    _BLOCK_TILE_M,
    _BLOCK_TILE_N,
    _CONV_CUDA,
    _CONV_HIP,
    _WMMA_DIM,
    ConvLaunchPlan,
    ConvShapeMeta,
    _m_total,
)

# --- Forward Template -------------------------------------------------------

CNN_FORWARD_TEMPLATE = Template(r'''
$hip_include

#define C_IN    $c_in
#define C_OUT   $c_out
#define F_H     $f_h
#define F_W     $f_w
#define STR_H   $str_h
#define STR_W   $str_w
#define PAD_H   $pad_h
#define PAD_W   $pad_w
#define H_IN    $h_in
#define W_IN    $w_in
#define H_OUT   $h_out
#define W_OUT   $w_out
#define K_TOTAL (F_H * F_W * C_IN)
#define HW_OUT  (H_OUT * W_OUT)
#define C_IN_FW (C_IN * F_W)

#define WMMA_M  $wmma_dim
#define WMMA_N  $wmma_dim
#define WMMA_K  $wmma_dim

#define BLOCK_TILE_M $block_tile_m
#define BLOCK_TILE_N $block_tile_n
#define WARPS_M (BLOCK_TILE_M / WMMA_M)
#define WARPS_N (BLOCK_TILE_N / WMMA_N)
#define WARP_SIZE $warp_size

using $wmma_ns::load_matrix_sync;
using $wmma_ns::store_matrix_sync;
using $wmma_ns::mma_sync;
using $wmma_ns::fill_fragment;
using $wmma_ns::fragment;

extern "C" __global__ void conv_forward_wmma(
    const half* __restrict__ input,        // [S, H_IN, W_IN, C_IN]
    const half* __restrict__ weight_fp16,  // [K_TOTAL, C_OUT]
    const float* __restrict__ bias,        // [C_OUT]
    float* __restrict__ output,            // [S, H_OUT, W_OUT, C_OUT]
    const long long M_total                // S * H_OUT * W_OUT
) {
    const long long m_tile_start = (long long)blockIdx.x * BLOCK_TILE_M;
    const int n_tile_start = blockIdx.y * BLOCK_TILE_N;

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    // Warps whose 16 columns lie past C_OUT would multiply zeros; skip them.
    const bool warp_live = warp_row < WARPS_M && warp_col < WARPS_N
                        && n_tile_start + warp_col * WMMA_N < C_OUT;

    __shared__ half As[BLOCK_TILE_M][WMMA_K + 8];
    __shared__ half Bs[WMMA_K][BLOCK_TILE_N + 8];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    const int num_k_tiles = (K_TOTAL + WMMA_K - 1) / WMMA_K;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        for (int idx = threadIdx.x; idx < BLOCK_TILE_M * WMMA_K; idx += blockDim.x) {
            const int local_m = idx / WMMA_K;
            const int local_k = idx % WMMA_K;
            const unsigned int m_global = (unsigned int)(m_tile_start + local_m);
            const int k_global = kt * WMMA_K + local_k;

            half val = __float2half(0.0f);
            if (m_global < (unsigned int)M_total && k_global < K_TOTAL) {
                const unsigned int s     = m_global / HW_OUT;
                const unsigned int rem_m = m_global % HW_OUT;
                const int h_out = (int)(rem_m / W_OUT);
                const int w_out = (int)(rem_m % W_OUT);

                const int c_in = k_global % C_IN;
                const int fw   = (k_global / C_IN) % F_W;
                const int fh   = k_global / C_IN_FW;

                const int h_in = h_out * STR_H - PAD_H + fh;
                const int w_in = w_out * STR_W - PAD_W + fw;

                if (h_in >= 0 && h_in < H_IN && w_in >= 0 && w_in < W_IN) {
                    const long long in_idx = (((long long)s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in;
                    val = input[in_idx];
                }
            }
            As[local_m][local_k] = val;
        }

        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x) {
            const int local_k = idx / BLOCK_TILE_N;
            const int local_n = idx % BLOCK_TILE_N;
            const int k_global = kt * WMMA_K + local_k;
            const int n_global = n_tile_start + local_n;

            half val = __float2half(0.0f);
            if (k_global < K_TOTAL && n_global < C_OUT) {
                val = weight_fp16[(long long)k_global * C_OUT + n_global];
            }
            Bs[local_k][local_n] = val;
        }

        __syncthreads();

        if (warp_live) {
            load_matrix_sync(a_frag, &As[warp_row * WMMA_M][0], WMMA_K + 8);
            load_matrix_sync(b_frag, &Bs[0][warp_col * WMMA_N], BLOCK_TILE_N + 8);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    __shared__ float Cs[BLOCK_TILE_M][BLOCK_TILE_N];
    if (warp_row < WARPS_M && warp_col < WARPS_N) {
        store_matrix_sync(&Cs[warp_row * WMMA_M][warp_col * WMMA_N], acc_frag,
                           BLOCK_TILE_N, $mem_layout);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < BLOCK_TILE_M * BLOCK_TILE_N; idx += blockDim.x) {
        const int local_m = idx / BLOCK_TILE_N;
        const int local_n = idx % BLOCK_TILE_N;
        const long long m_global = m_tile_start + local_m;
        const int n_global = n_tile_start + local_n;

        if (m_global < M_total && n_global < C_OUT) {
            output[m_global * C_OUT + n_global] = Cs[local_m][local_n] + bias[n_global];
        }
    }
}
''')

_CONV_FORWARD_KERNEL_CACHE = {}


def get_compiled_forward_conv_kernel(shape_meta: ConvShapeMeta):
    key = shape_meta.cache_key
    if key in _CONV_FORWARD_KERNEL_CACHE:
        return _CONV_FORWARD_KERNEL_CACHE[key]

    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA
    source = CNN_FORWARD_TEMPLATE.substitute(
        c_in=shape_meta.C_in, c_out=shape_meta.C_out,
        f_h=shape_meta.fH, f_w=shape_meta.fW,
        str_h=shape_meta.sH, str_w=shape_meta.sW,
        pad_h=shape_meta.pad_h, pad_w=shape_meta.pad_w,
        h_in=shape_meta.H_in, w_in=shape_meta.W_in,
        h_out=shape_meta.H_out, w_out=shape_meta.W_out,
        wmma_dim=_WMMA_DIM,
        block_tile_m=_BLOCK_TILE_M, block_tile_n=_BLOCK_TILE_N,
        **vendor,
    )

    options = ('-std=c++17',)
    if shape_meta.is_hip:
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        options = options + (f'-I{rocm_path}/include',)

    try:
        module = config.cp.RawModule(code=source, options=options)
        kernel = module.get_function('conv_forward_wmma')
    except (config.cp.cuda.compiler.CompileException, config.cp.cuda.driver.CUDADriverError) as e:
        warnings.warn(
            f"[aether] conv_kernel: matrix-core conv compile failed for "
            f"shape {key}, this Conv layer will fall back to the scalar "
            f"path: {e}"
        )
        _CONV_FORWARD_KERNEL_CACHE[key] = None
        return None

    threads_per_block = (_BLOCK_TILE_M // _WMMA_DIM) * (_BLOCK_TILE_N // _WMMA_DIM) * vendor['warp_size']
    launch_meta = dict(
        block=(threads_per_block,),
        block_tile_m=_BLOCK_TILE_M,
        block_tile_n=_BLOCK_TILE_N,
    )

    _CONV_FORWARD_KERNEL_CACHE[key] = (kernel, launch_meta)
    return _CONV_FORWARD_KERNEL_CACHE[key]


def plan_forward_launch(shape_meta: ConvShapeMeta, S: int):
    """conv_forward_wmma(input_fp16, weight_fp16, bias, output) -> plan."""
    compiled = get_compiled_forward_conv_kernel(shape_meta)
    if compiled is None:
        return None
    kernel, launch = compiled

    M_total = _m_total(shape_meta, S)
    grid = (
        int(-(-M_total // launch["block_tile_m"])),
        int(-(-shape_meta.C_out // launch["block_tile_n"])),
    )
    return ConvLaunchPlan(kernel, grid, launch["block"], M_total, shape_meta)

