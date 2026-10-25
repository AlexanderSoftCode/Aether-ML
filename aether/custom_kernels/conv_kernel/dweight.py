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

# LDS padding for the dweight kernel's staging buffers, in halves. Tuned
# empirically against the column-major A-fragment read pattern; re-sweep
# before changing.
_DWEIGHT_A_PAD = 8
_DWEIGHT_B_PAD = 8

# No stream-K algorithm because I said so
def choose_dweight_split_k(num_m_tiles, grid_x, grid_y, threads_per_block):
    """Pick the number of split-K slices for the dweight grid.

    The grid should fill every resident block slot once: fewer blocks leave
    the GPU idle, and splitting past one wave only adds per-block epilogues
    and a ragged tail. Threads are the binding occupancy limit for this
    kernel (LDS and registers allow more blocks per MP).
    """
    mp_count, max_threads_per_mp = config.get_device_multiprocessor_limits()
    slots = mp_count * max(1, max_threads_per_mp // threads_per_block)
    b_2d = max(1, grid_x * grid_y)
    return max(1, min(num_m_tiles, slots // b_2d))


CNN_BACKWARD_DWEIGHT_TEMPLATE = Template(r'''
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
#define THREADS (WARPS_M * WARPS_N * WARP_SIZE)

// LDS padding. As is indexed [pixel][filter] so its leading dimension is
// BLOCK_TILE_M; A_PAD spreads the column-major fragment reads across banks.
#define A_PAD $a_pad
#define B_PAD $b_pad
#define A_LD (BLOCK_TILE_M + A_PAD)

using $wmma_ns::load_matrix_sync;
using $wmma_ns::store_matrix_sync;
using $wmma_ns::mma_sync;
using $wmma_ns::fill_fragment;
using $wmma_ns::fragment;

// bias_acc is sized by the B-tile strided-loop trip count; keep them in sync.
#define BIAS_ITEMS (((WMMA_K * BLOCK_TILE_N) + THREADS - 1) / THREADS)

// The A-gather hoists its filter-index decomposition out of the reduction
// loop, which is only valid if a lane's filter index never changes as the
// strided staging loop advances by THREADS.
static_assert(THREADS % BLOCK_TILE_M == 0, "THREADS must be a multiple of BLOCK_TILE_M");
static_assert(THREADS >= BLOCK_TILE_N, "bias flush assumes one thread per N column");

extern "C" __global__ void __launch_bounds__(THREADS) conv_backward_dweight_wmma(
    const half* __restrict__ input,        // [S, H_IN, W_IN, C_IN]
    const half* __restrict__ dvalues,      // [S, H_OUT, W_OUT, C_OUT]
    float* __restrict__ dweights,          // [K_TOTAL, C_OUT]
    float* __restrict__ dbiases,           // [C_OUT]
    const long long M_total                // S * H_OUT * W_OUT
) {
    // Implicit GEMM mapping: dW = X^T [K_TOTAL, M_total] * dY [M_total, C_OUT]
    //
    // X^T is produced by the *fragment layout*, not by a transposed gather:
    // As is staged [pixel][filter] (the orientation that makes the im2col
    // gather coalesced along C_IN) and read back with a column-major A
    // fragment, which reinterprets element (i,j) as As[j][i] == X^T.
    const int k_tile_start = blockIdx.x * BLOCK_TILE_M;
    const int n_tile_start = blockIdx.y * BLOCK_TILE_N;

    const int warp_id  = threadIdx.x / WARP_SIZE;
    const int warp_row = warp_id / WARPS_N;
    const int warp_col = warp_id % WARPS_N;
    // Warps whose 16x16 tile lies past K_TOTAL or C_OUT would multiply
    // zeros (e.g. C_OUT=32 or K_TOTAL=27 leave half the 64x64 tile empty).
    const bool warp_live = warp_row < WARPS_M && warp_col < WARPS_N
                        && k_tile_start + warp_row * WMMA_M < K_TOTAL
                        && n_tile_start + warp_col * WMMA_N < C_OUT;

    __shared__ half As[WMMA_K][A_LD];
    __shared__ half Bs[WMMA_K][BLOCK_TILE_N + B_PAD];
    __shared__ float s_bias[BLOCK_TILE_N];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a_col> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    float bias_acc[BIAS_ITEMS];
    #pragma unroll
    for (int i = 0; i < BIAS_ITEMS; ++i) bias_acc[i] = 0.0f;

    if (blockIdx.x == 0 && threadIdx.x < BLOCK_TILE_N) {
        s_bias[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    const int a_lane_k  = threadIdx.x % BLOCK_TILE_M;
    const int a_lane_m0 = threadIdx.x / BLOCK_TILE_M;
    const int a_rows_per_pass = THREADS / BLOCK_TILE_M;

    const int k_global_a = k_tile_start + a_lane_k;
    const bool a_k_valid = (k_global_a < K_TOTAL);
    const int c_in_a = a_k_valid ? (k_global_a % C_IN)        : 0;
    const int fw_a   = a_k_valid ? ((k_global_a / C_IN) % F_W) : 0;
    const int fh_a   = a_k_valid ? (k_global_a / C_IN_FW)      : 0;

    // Split-K reduction distribution across gridDim.z
    const int num_m_tiles = (int)((M_total + WMMA_K - 1) / WMMA_K);
    const int tiles_per_slice = (num_m_tiles + gridDim.z - 1) / gridDim.z;
    const int mt_start = blockIdx.z * tiles_per_slice;
    const int mt_end = (mt_start + tiles_per_slice < num_m_tiles) ? (mt_start + tiles_per_slice) : num_m_tiles;

    for (int mt = mt_start; mt < mt_end; ++mt) {

        // --- A tile: coalesced im2col gather, staged [pixel][filter] ---
        // Consecutive lanes walk consecutive filter indices, hence consecutive
        // c_in, hence contiguous `input` addresses. The LDS write is likewise
        // contiguous and bank-conflict free.
        #pragma unroll
        for (int local_m = a_lane_m0; local_m < WMMA_K; local_m += a_rows_per_pass) {
            const unsigned int m_global = (unsigned int)mt * WMMA_K + local_m;

            half val = __float2half(0.0f);
            if (a_k_valid && m_global < (unsigned int)M_total) {
                const unsigned int s     = m_global / HW_OUT;
                const unsigned int rem_m = m_global % HW_OUT;
                const int h_out = (int)(rem_m / W_OUT);
                const int w_out = (int)(rem_m % W_OUT);

                const int h_in = h_out * STR_H - PAD_H + fh_a;
                const int w_in = w_out * STR_W - PAD_W + fw_a;

                if (h_in >= 0 && h_in < H_IN && w_in >= 0 && w_in < W_IN) {
                    const long long in_idx = (((long long)s * H_IN + h_in) * W_IN + w_in) * C_IN + c_in_a;
                    val = input[in_idx];
                }
            }
            As[local_m][a_lane_k] = val;
        }

        // --- B tile: Load dY & register-accumulate bias ---
        if (blockIdx.x == 0) {
            int item = 0;
            for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x, ++item) {
                const int local_m = idx / BLOCK_TILE_N;
                const int local_n = idx % BLOCK_TILE_N;
                const unsigned int m_global = (unsigned int)mt * WMMA_K + local_m;
                const int n_global = n_tile_start + local_n;

                half val = __float2half(0.0f);
                if (m_global < (unsigned int)M_total && n_global < C_OUT) {
                    val = dvalues[(long long)m_global * C_OUT + n_global];
                    bias_acc[item] += __half2float(val);
                }
                Bs[local_m][local_n] = val;
            }
        } else {
            for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x) {
                const int local_m = idx / BLOCK_TILE_N;
                const int local_n = idx % BLOCK_TILE_N;
                const unsigned int m_global = (unsigned int)mt * WMMA_K + local_m;
                const int n_global = n_tile_start + local_n;

                half val = __float2half(0.0f);
                if (m_global < (unsigned int)M_total && n_global < C_OUT) {
                    val = dvalues[(long long)m_global * C_OUT + n_global];
                }
                Bs[local_m][local_n] = val;
            }
        }

        __syncthreads();

        if (warp_live) {
            // col_major: element (i, j) == ptr[i + j * ldm] == As[pixel j][filter i]
            load_matrix_sync(a_frag, &As[0][warp_row * WMMA_M], A_LD);
            load_matrix_sync(b_frag, &Bs[0][warp_col * WMMA_N], BLOCK_TILE_N + B_PAD);
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    __shared__ float Cs[BLOCK_TILE_M][BLOCK_TILE_N];
    if (warp_row < WARPS_M && warp_col < WARPS_N) {
        store_matrix_sync(&Cs[warp_row * WMMA_M][warp_col * WMMA_N], acc_frag, BLOCK_TILE_N, $mem_layout);
    }
    __syncthreads();

    // --- Epilogue: dweights global store / accumulation ---
    for (int idx = threadIdx.x; idx < BLOCK_TILE_M * BLOCK_TILE_N; idx += blockDim.x) {
        const int local_k = idx / BLOCK_TILE_N;
        const int local_n = idx % BLOCK_TILE_N;
        const int k_global = k_tile_start + local_k;
        const int n_global = n_tile_start + local_n;

        if (k_global < K_TOTAL && n_global < C_OUT) {
            const float val = Cs[local_k][local_n];
            if (gridDim.z > 1) {
                atomicAdd(&dweights[k_global * C_OUT + n_global], val);
            } else {
                dweights[k_global * C_OUT + n_global] = val;
            }
        }
    }

    // --- Epilogue: Local bias flush to shared memory ---
    if (blockIdx.x == 0) {
        int item = 0;
        for (int idx = threadIdx.x; idx < WMMA_K * BLOCK_TILE_N; idx += blockDim.x, ++item) {
            const int local_n = idx % BLOCK_TILE_N;
            if (bias_acc[item] != 0.0f) {
                atomicAdd(&s_bias[local_n], bias_acc[item]);
            }
        }
    }
    __syncthreads();

    // --- Epilogue: Atomic global write for dbiases per slice ---
    if (blockIdx.x == 0 && threadIdx.x < BLOCK_TILE_N) {
        const int n_global = n_tile_start + threadIdx.x;
        if (n_global < C_OUT) {
            atomicAdd(&dbiases[n_global], s_bias[threadIdx.x]);
        }
    }
}
''')

_CONV_BACKWARD_DWEIGHT_CACHE = {}


def get_compiled_backward_dweight_kernel(shape_meta: ConvShapeMeta):
    key = shape_meta.cache_key
    if key in _CONV_BACKWARD_DWEIGHT_CACHE:
        return _CONV_BACKWARD_DWEIGHT_CACHE[key]

    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA
    block_tile_m, block_tile_n = _BLOCK_TILE_M, _BLOCK_TILE_N
    source = CNN_BACKWARD_DWEIGHT_TEMPLATE.substitute(
        c_in=shape_meta.C_in, c_out=shape_meta.C_out,
        f_h=shape_meta.fH, f_w=shape_meta.fW,
        str_h=shape_meta.sH, str_w=shape_meta.sW,
        pad_h=shape_meta.pad_h, pad_w=shape_meta.pad_w,
        h_in=shape_meta.H_in, w_in=shape_meta.W_in,
        h_out=shape_meta.H_out, w_out=shape_meta.W_out,
        wmma_dim=_WMMA_DIM,
        block_tile_m=block_tile_m, block_tile_n=block_tile_n,
        a_pad=_DWEIGHT_A_PAD, b_pad=_DWEIGHT_B_PAD,
        **vendor,
    )

    options = ('-std=c++17',)
    if shape_meta.is_hip:
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        options = options + (f'-I{rocm_path}/include',)

    try:
        module = config.cp.RawModule(code=source, options=options)
        kernel = module.get_function('conv_backward_dweight_wmma')
    except (config.cp.cuda.compiler.CompileException, config.cp.cuda.driver.CUDADriverError) as e:
        warnings.warn(f"[aether] conv_kernel: matrix-core dweight compile failed: {e}")
        _CONV_BACKWARD_DWEIGHT_CACHE[key] = None
        return None

    threads_per_block = (block_tile_m // _WMMA_DIM) * (block_tile_n // _WMMA_DIM) * vendor['warp_size']
    launch_meta = dict(
        block=(threads_per_block,),
        block_tile_m=block_tile_m,
        block_tile_n=block_tile_n,
    )

    _CONV_BACKWARD_DWEIGHT_CACHE[key] = (kernel, launch_meta)
    return _CONV_BACKWARD_DWEIGHT_CACHE[key]


def plan_dweight_launch(shape_meta: ConvShapeMeta, S: int):
    """conv_backward_dweight_wmma(input_fp16, dvalues_fp16, dweights, dbiases).

    Note the 3D grid: gridDim.z carries the split-K slices that make this
    kernel fast. dweights/dbiases must be zeroed by the caller whenever
    gridDim.z > 1, since the epilogue accumulates with atomicAdd.
    """
    compiled = get_compiled_backward_dweight_kernel(shape_meta)
    if compiled is None:
        return None
    kernel, launch = compiled

    M_total = _m_total(shape_meta, S)
    grid_x = int(-(-shape_meta.K_total // launch["block_tile_m"]))
    grid_y = int(-(-shape_meta.C_out // launch["block_tile_n"]))
    num_m_tiles = int(-(-M_total // _WMMA_DIM))
    split_k = choose_dweight_split_k(num_m_tiles, grid_x, grid_y, launch["block"][0])

    return ConvLaunchPlan(
        kernel, (grid_x, grid_y, split_k), launch["block"], M_total, shape_meta
    )

