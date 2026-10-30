import warnings
from string import Template

import aether.config as config
from aether.custom_kernels.conv_kernel.common import (
    _BLOCK_TILE_M,
    _COMPILE_OPTIONS,
    _CONV_CUDA,
    _CONV_HIP,
    _WMMA_DIM,
    ConvLaunchPlan,
    ConvShapeMeta,
    _m_total,
    choose_block_tile_n,
    threads_per_block,
)

CNN_BACKWARD_DINPUT_TEMPLATE = Template(r'''
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

#define TAPS_H      ((F_H + STR_H - 1) / STR_H)
#define TAPS_W      ((F_W + STR_W - 1) / STR_W)
#define K_GATHER    (TAPS_H * TAPS_W * C_OUT)
#define C_OUT_TAPSW (C_OUT * TAPS_W)
#define HW_OUT      (H_OUT * W_OUT)

#define WMMA_M  $wmma_dim
#define WMMA_N  $wmma_dim
#define WMMA_K  $wmma_dim

#define BLOCK_TILE_M $block_tile_m
#define BLOCK_TILE_N $block_tile_n
#define WARPS_M (BLOCK_TILE_M / WMMA_M)
#define WARPS_N (BLOCK_TILE_N / WMMA_N)
#define WARP_SIZE $warp_size
#define THREADS (WARPS_M * WARPS_N * WARP_SIZE)

// 16-byte staging of A (dvalues): a lane moves 8 consecutive c_out (one
// uint4) of one tap when C_OUT keeps every 8-aligned k chunk inside one tap.
// Otherwise A is staged one element per lane.
#define VEC 8
#define VEC_A ($vectorize && C_OUT % VEC == 0)
#define A_VEC_THREADS (BLOCK_TILE_M * WMMA_K / VEC)
static_assert(A_VEC_THREADS <= THREADS, "vector staging needs a lane per A chunk");

// Trip count of the A staging loop. Each lane keeps one hoisted addressing
// triple per pass, so this has to be a compile-time constant.
#define A_PASSES ((BLOCK_TILE_M * WMMA_K + THREADS - 1) / THREADS)

using $wmma_ns::load_matrix_sync;
using $wmma_ns::store_matrix_sync;
using $wmma_ns::mma_sync;
using $wmma_ns::fill_fragment;
using $wmma_ns::fragment;

extern "C" __global__ void __launch_bounds__(THREADS) conv_backward_dinput_wmma(
    const half* __restrict__ dvalues,      // [S, H_OUT, W_OUT, C_OUT]
    const half* __restrict__ weight_fp16,  // [F_H*F_W*C_IN, C_OUT]
    float* __restrict__ dinputs,           // [S, H_IN, W_IN, C_IN]
    const long long M_total                // S * H_OUT * W_OUT
) {
    const int u = blockIdx.z / STR_W;
    const int v = blockIdx.z % STR_W;

    const int r_h     = (u + PAD_H) % STR_H;
    const int r_w     = (v + PAD_W) % STR_W;
    const int base_qh = (u + PAD_H) / STR_H;
    const int base_qw = (v + PAD_W) / STR_W;
    const int NH = (H_IN - u + STR_H - 1) / STR_H;
    const int NW = (W_IN - v + STR_W - 1) / STR_W;

    // Substituting fh = r_h + t_h*STR_H into the constraint collapses the
    // output coordinate to a subtraction: h_out = base_qh + j - t_h.
    const int HW_CLASS = NH * NW;
    // S is not a compile-time constant -- ConvShapeMeta is deliberately
    // batch-independent so one compiled module serves every batch size.
    // M_total is exactly S * HW_OUT, so the division recovers S exactly.
    const long long M_class = (M_total / HW_OUT) * HW_CLASS;

    const long long m_tile_start = (long long)blockIdx.x * BLOCK_TILE_M;
    if (m_tile_start >= M_class) return;

    const int n_tile_start = blockIdx.y * BLOCK_TILE_N;

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    // Warps whose 16 columns lie past C_IN would multiply zeros; skip them.
    const bool warp_live = warp_row < WARPS_M && warp_col < WARPS_N
                        && n_tile_start + warp_col * WMMA_N < C_IN;

    __shared__ alignas(32) half As[BLOCK_TILE_M][WMMA_K + 8];
    __shared__ alignas(32) half Bs[WMMA_K][BLOCK_TILE_N + 8];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

#if VEC_A
    // A lane's pixel is fixed for the whole K loop, so decompose it once.
    const bool a_lane = threadIdx.x < A_VEC_THREADS;
    const int a_m  = threadIdx.x / (WMMA_K / VEC);
    const int a_k8 = (threadIdx.x % (WMMA_K / VEC)) * VEC;
    const long long a_m_global = m_tile_start + a_m;
    const bool a_valid = a_lane && a_m_global < M_class;
    int a_s = 0, a_hob = 0, a_wob = 0;
    if (a_valid) {
        a_s = (int)(a_m_global / HW_CLASS);
        const int rem = (int)(a_m_global % HW_CLASS);
        a_hob = base_qh + rem / NW;
        a_wob = base_qw + rem % NW;
    }
#else
    int a_local_m[A_PASSES];
    int a_valid[A_PASSES];
    int a_s[A_PASSES];
    int a_hob[A_PASSES];   // base_qh + j
    int a_wob[A_PASSES];   // base_qw + i

    #pragma unroll
    for (int p = 0; p < A_PASSES; ++p) {
        const int idx = threadIdx.x + p * THREADS;
        const int local_m = idx / WMMA_K;
        const long long m_global = m_tile_start + local_m;

        const int in_range = (idx < BLOCK_TILE_M * WMMA_K) && (m_global < M_class);
        a_local_m[p] = local_m;
        a_valid[p]   = in_range;
        a_s[p] = 0; a_hob[p] = 0; a_wob[p] = 0;

        if (in_range) {
            a_s[p] = (int)(m_global / HW_CLASS);
            const int rem = (int)(m_global % HW_CLASS);
            a_hob[p] = base_qh + rem / NW;
            a_wob[p] = base_qw + rem % NW;
        }
    }
#endif

    const int num_k_tiles = (K_GATHER + WMMA_K - 1) / WMMA_K;

    for (int kt = 0; kt < num_k_tiles; ++kt) {

#if VEC_A
        if (a_lane) {
            // K_GATHER is a multiple of 8, so the chunk is all in range or all out.
            const int k_global = kt * WMMA_K + a_k8;
            uint4 val = {0u, 0u, 0u, 0u};
            if (a_valid && k_global < K_GATHER) {
                const int c_out = k_global % C_OUT;
                const int t_w   = (k_global / C_OUT) % TAPS_W;
                const int t_h   = k_global / C_OUT_TAPSW;

                const int fh    = r_h + t_h * STR_H;
                const int fw    = r_w + t_w * STR_W;
                const int h_out = a_hob - t_h;
                const int w_out = a_wob - t_w;

                if (fh < F_H && fw < F_W &&
                    h_out >= 0 && h_out < H_OUT &&
                    w_out >= 0 && w_out < W_OUT) {
                    const long long dv_idx =
                        (((long long)a_s * H_OUT + h_out) * W_OUT + w_out) * C_OUT + c_out;
                    val = *reinterpret_cast<const uint4*>(&dvalues[dv_idx]);
                }
            }
            *reinterpret_cast<uint4*>(&As[a_m][a_k8]) = val;
        }
#else
        #pragma unroll
        for (int p = 0; p < A_PASSES; ++p) {
            const int idx = threadIdx.x + p * THREADS;
            if (idx >= BLOCK_TILE_M * WMMA_K) break;

            const int local_k = idx % WMMA_K;
            const int k_global = kt * WMMA_K + local_k;

            half val = __float2half(0.0f);
            if (a_valid[p] && k_global < K_GATHER) {
                const int c_out = k_global % C_OUT;
                const int t_w   = (k_global / C_OUT) % TAPS_W;
                const int t_h   = k_global / C_OUT_TAPSW;

                const int fh    = r_h + t_h * STR_H;
                const int fw    = r_w + t_w * STR_W;
                const int h_out = a_hob[p] - t_h;
                const int w_out = a_wob[p] - t_w;

                if (fh < F_H && fw < F_W &&
                    h_out >= 0 && h_out < H_OUT &&
                    w_out >= 0 && w_out < W_OUT) {
                    const long long dv_idx =
                        (((long long)a_s[p] * H_OUT + h_out) * W_OUT + w_out) * C_OUT + c_out;
                    val = dvalues[dv_idx];
                }
            }
            As[a_local_m[p]][local_k] = val;
        }
#endif

        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x) {
            const int local_k = idx % WMMA_K;
            const int local_n = idx / WMMA_K;
            const int k_global = kt * WMMA_K + local_k;
            const int n_global = n_tile_start + local_n;

            half val = __float2half(0.0f);
            if (k_global < K_GATHER && n_global < C_IN) {
                const int c_out = k_global % C_OUT;
                const int t_w   = (k_global / C_OUT) % TAPS_W;
                const int t_h   = k_global / C_OUT_TAPSW;

                const int fh = r_h + t_h * STR_H;
                const int fw = r_w + t_w * STR_W;

                if (fh < F_H && fw < F_W) {
                    const long long w_idx =
                        (((long long)fh * F_W + fw) * C_IN + n_global) * C_OUT + c_out;
                    val = weight_fp16[w_idx];
                }
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
        const int c_in = n_tile_start + local_n;

        if (m_global < M_class && c_in < C_IN) {
            const int s    = (int)(m_global / HW_CLASS);
            const int rem  = (int)(m_global % HW_CLASS);
            const int h_in = u + (rem / NW) * STR_H;
            const int w_in = v + (rem % NW) * STR_W;

            const long long in_idx =
                (((long long)s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in;
            dinputs[in_idx] = Cs[local_m][local_n];
        }
    }
}
''')

_CONV_BACKWARD_DINPUT_CACHE = {}


def _dinput_tiles(shape_meta: ConvShapeMeta):
    # dX = dY * W^T: the GEMM N axis is C_in.
    return _BLOCK_TILE_M, choose_block_tile_n(shape_meta.C_in)


def render_dinput_source(shape_meta: ConvShapeMeta, vectorize=True):
    """CUDA/HIP source for one shape. vectorize=False forces the scalar staging."""
    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA
    block_tile_m, block_tile_n = _dinput_tiles(shape_meta)
    return CNN_BACKWARD_DINPUT_TEMPLATE.substitute(
        c_in=shape_meta.C_in, c_out=shape_meta.C_out,
        f_h=shape_meta.fH, f_w=shape_meta.fW,
        str_h=shape_meta.sH, str_w=shape_meta.sW,
        pad_h=shape_meta.pad_h, pad_w=shape_meta.pad_w,
        h_in=shape_meta.H_in, w_in=shape_meta.W_in,
        h_out=shape_meta.H_out, w_out=shape_meta.W_out,
        wmma_dim=_WMMA_DIM,
        block_tile_m=block_tile_m, block_tile_n=block_tile_n,
        vectorize=int(bool(vectorize)),
        **vendor,
    )


def get_compiled_backward_dinput_kernel(shape_meta: ConvShapeMeta):
    key = shape_meta.cache_key
    if key in _CONV_BACKWARD_DINPUT_CACHE:
        return _CONV_BACKWARD_DINPUT_CACHE[key]

    source = render_dinput_source(shape_meta)

    try:
        module = config.cp.RawModule(code=source, options=_COMPILE_OPTIONS)
        kernel = module.get_function('conv_backward_dinput_wmma')
    except (config.cp.cuda.compiler.CompileException, config.cp.cuda.driver.CUDADriverError) as e:
        warnings.warn(f"[aether] conv_kernel: matrix-core dinput compile failed: {e}")
        _CONV_BACKWARD_DINPUT_CACHE[key] = None
        return None

    block_tile_m, block_tile_n = _dinput_tiles(shape_meta)
    launch_meta = dict(
        block=(threads_per_block(block_tile_m, block_tile_n),),
        block_tile_m=block_tile_m,
        block_tile_n=block_tile_n,
    )

    _CONV_BACKWARD_DINPUT_CACHE[key] = (kernel, launch_meta)
    return _CONV_BACKWARD_DINPUT_CACHE[key]


def plan_dinput_launch(shape_meta: ConvShapeMeta, S: int):
    """conv_backward_dinput_wmma(dvalues_fp16, weight_fp16, dinputs).

    gridDim.z carries the residue class (h_in % sH, w_in % sW). The class has
    to be block-uniform because the B operand depends on it, and classes differ
    in size by at most one row/column, so the grid is sized for the largest and
    surplus blocks exit immediately. At stride 1 there is exactly one class.
    """
    compiled = get_compiled_backward_dinput_kernel(shape_meta)
    if compiled is None:
        return None
    kernel, launch = compiled

    M_total = _m_total(shape_meta, S)

    # Class u=0 always attains the maximum row count, likewise v=0 for columns.
    nh_max = -(-shape_meta.H_in // shape_meta.sH)
    nw_max = -(-shape_meta.W_in // shape_meta.sW)
    max_class_m = S * nh_max * nw_max

    grid = (
        int(-(-max_class_m // launch["block_tile_m"])),
        int(-(-shape_meta.C_in // launch["block_tile_n"])),
        int(shape_meta.sH * shape_meta.sW),
    )
    return ConvLaunchPlan(kernel, grid, launch["block"], M_total, shape_meta)
