import os
import warnings
from dataclasses import dataclass
from string import Template

import numpy as np

import aether.config as config


def get_is_conv_gpu_available():
    """Check if GPU supports optimized tensor-core convolution kernels."""
    return config.get_tensor_core_capable()


# --- Shape metadata / cache key --------------------------------------------

@dataclass(frozen=True)
class ConvShapeMeta:
    H_in: int
    W_in: int
    C_in: int
    C_out: int
    fH: int
    fW: int
    sH: int
    sW: int
    pad_h: int
    pad_w: int
    H_out: int
    W_out: int
    is_hip: bool

    @property
    def K_total(self):
        return self.fH * self.fW * self.C_in

    # pad_h/pad_w are the top/left pads the kernels offset by; the bottom/right
    # pads are never materialized there, the kernels bounds-check them away.
    @property
    def pad_bottom(self):
        return (self.H_out - 1) * self.sH + self.fH - self.H_in - self.pad_h

    @property
    def pad_right(self):
        return (self.W_out - 1) * self.sW + self.fW - self.W_in - self.pad_w

    @property
    def cache_key(self):
        return (
            self.H_in, self.W_in, self.C_in, self.C_out,
            self.fH, self.fW, self.sH, self.sW,
            # H_out/W_out are not implied by the rest: 'same' and 'valid' can
            # share a pad yet differ in output extent.
            self.pad_h, self.pad_w, self.H_out, self.W_out, self.is_hip,
        )

    @classmethod
    def create(cls, spatial_shape, C_out, filter_size, stride, padding, is_hip=None):
        """Derive padding and output extents from a layer's configuration.

        This is the only place the conv spatial geometry is computed. Callers
        -- layers, benchmarks, profilers -- must go through here rather than
        rederiving it, so a script can never silently disagree with the
        kernel about what shape it is running.
        """
        H_in, W_in, C_in = spatial_shape
        fH, fW = filter_size
        sH, sW = stride

        if padding == "same":
            # TF convention: total pad covers ceil(in / stride) windows; an odd
            # total puts the extra pixel on the bottom/right.
            H_out, W_out = -(-H_in // sH), -(-W_in // sW)
            pad_h = max((H_out - 1) * sH + fH - H_in, 0) // 2
            pad_w = max((W_out - 1) * sW + fW - W_in, 0) // 2
        elif padding == "valid":
            H_out, W_out = (H_in - fH) // sH + 1, (W_in - fW) // sW + 1
            pad_h, pad_w = 0, 0
        else:
            raise ValueError(
                f"Unsupported padding mode '{padding}'. Use 'same' or 'valid'."
            )

        return cls(
            H_in=H_in, W_in=W_in, C_in=C_in, C_out=C_out,
            fH=fH, fW=fW, sH=sH, sW=sW, pad_h=pad_h, pad_w=pad_w,
            H_out=H_out, W_out=W_out,
            is_hip=config.IS_HIP if is_hip is None else is_hip,
        )


_BLOCK_TILE_M = 64
_BLOCK_TILE_N = 64
_WMMA_DIM = 16
_WARP_SIZE = 32

# LDS padding for the dweight kernel's staging buffers, in halves. Tuned
# empirically against the column-major A-fragment read pattern via
# examples/cifar10/dweight_bench.py -- see the sweep there before changing.
_DWEIGHT_A_PAD = 8
_DWEIGHT_B_PAD = 8


_MP_COUNT = None


def get_device_multiprocessor_count():
    """Memoized MP count, used to size the dweight split-K grid."""
    global _MP_COUNT
    if _MP_COUNT is None:
        try:
            device = config.cp.cuda.Device()
            props = config.cp.cuda.runtime.getDeviceProperties(device.id)
            _MP_COUNT = int(props['multiProcessorCount'])
        except config.cp.cuda.runtime.CUDARuntimeError:
            _MP_COUNT = 32
    return _MP_COUNT


def choose_dweight_split_k(num_m_tiles, grid_x, grid_y):
    """Pick the number of split-K slices for the dweight grid.
    """
    mp_count = get_device_multiprocessor_count()
    b_2d = max(1, grid_x * grid_y)
    target_blocks = (4 if b_2d == 1 else 2) * mp_count
    return max(1, min(num_m_tiles, target_blocks // b_2d))


# --- Vendor substitution dictionaries --------------------------------------

_CONV_CUDA = dict(
    hip_include="#include <mma.h>\n#include <cuda_fp16.h>",
    wmma_ns="nvcuda::wmma",
    warp_size=_WARP_SIZE,
    frag_layout_a="nvcuda::wmma::row_major",
    frag_layout_a_col="nvcuda::wmma::col_major",
    frag_layout_b="nvcuda::wmma::row_major",
    mem_layout="nvcuda::wmma::mem_row_major",
    matrix_a_tag="nvcuda::wmma::matrix_a",
    matrix_b_tag="nvcuda::wmma::matrix_b",
    accumulator_tag="nvcuda::wmma::accumulator",
)

_CONV_HIP = dict(
    hip_include="#include <hip/hip_fp16.h>\n#include <rocwmma/rocwmma.hpp>",
    wmma_ns="rocwmma",
    warp_size=_WARP_SIZE,
    frag_layout_a="rocwmma::row_major",
    frag_layout_a_col="rocwmma::col_major",
    frag_layout_b="rocwmma::row_major",
    mem_layout="rocwmma::mem_row_major",
    matrix_a_tag="rocwmma::matrix_a",
    matrix_b_tag="rocwmma::matrix_b",
    accumulator_tag="rocwmma::accumulator",
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

        if (warp_row < WARPS_M && warp_col < WARPS_N) {
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

# --- Backward Templates -----------------------------------------------------

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

        if (warp_row < WARPS_M && warp_col < WARPS_N) {
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

    __shared__ half As[BLOCK_TILE_M][WMMA_K + 8];
    __shared__ half Bs[WMMA_K][BLOCK_TILE_N + 8];

    fragment<$matrix_a_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_a> a_frag;
    fragment<$matrix_b_tag, WMMA_M, WMMA_N, WMMA_K, half, $frag_layout_b> b_frag;
    fragment<$accumulator_tag, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

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

    const int num_k_tiles = (K_GATHER + WMMA_K - 1) / WMMA_K;

    for (int kt = 0; kt < num_k_tiles; ++kt) {

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

        if (warp_row < WARPS_M && warp_col < WARPS_N) {
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

# --- Kernel Caching and Dispatch --------------------------------------------

_CONV_FORWARD_KERNEL_CACHE = {}
_CONV_BACKWARD_DWEIGHT_CACHE = {}
_CONV_BACKWARD_DINPUT_CACHE = {}


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


def get_compiled_backward_dinput_kernel(shape_meta: ConvShapeMeta):
    key = shape_meta.cache_key
    if key in _CONV_BACKWARD_DINPUT_CACHE:
        return _CONV_BACKWARD_DINPUT_CACHE[key]

    vendor = _CONV_HIP if shape_meta.is_hip else _CONV_CUDA
    source = CNN_BACKWARD_DINPUT_TEMPLATE.substitute(
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
        kernel = module.get_function('conv_backward_dinput_wmma')
    except (config.cp.cuda.compiler.CompileException, config.cp.cuda.driver.CUDADriverError) as e:
        warnings.warn(f"[aether] conv_kernel: matrix-core dinput compile failed: {e}")
        _CONV_BACKWARD_DINPUT_CACHE[key] = None
        return None

    threads_per_block = (_BLOCK_TILE_M // _WMMA_DIM) * (_BLOCK_TILE_N // _WMMA_DIM) * vendor['warp_size']
    launch_meta = dict(
        block=(threads_per_block,),
        block_tile_m=_BLOCK_TILE_M,
        block_tile_n=_BLOCK_TILE_N,
    )

    _CONV_BACKWARD_DINPUT_CACHE[key] = (kernel, launch_meta)
    return _CONV_BACKWARD_DINPUT_CACHE[key]

# --- Launch planning --------------------------------------------------------

@dataclass(frozen=True)
class ConvLaunchPlan:
    """A compiled kernel bound to a concrete grid, block and M_total."""
    kernel: object
    grid: tuple
    block: tuple
    M_total: object
    shape_meta: ConvShapeMeta

    def __call__(self, *tensors):
        """Launch. Every conv kernel takes M_total as its trailing argument."""
        self.kernel(self.grid, self.block, (*tensors, self.M_total))

    @property
    def num_blocks(self):
        n = 1
        for d in self.grid:
            n *= d
        return n


def _m_total(shape_meta: ConvShapeMeta, S: int):
    return np.int64(S * shape_meta.H_out * shape_meta.W_out)


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
    split_k = choose_dweight_split_k(num_m_tiles, grid_x, grid_y)

    return ConvLaunchPlan(
        kernel, (grid_x, grid_y, split_k), launch["block"], M_total, shape_meta
    )


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