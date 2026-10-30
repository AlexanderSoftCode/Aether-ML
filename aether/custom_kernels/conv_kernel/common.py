from dataclasses import dataclass

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
        """Derive padding and output extents from a layer's configuration."""
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


def choose_block_tile_n(n):
    """Block tile width on the GEMM N axis (C_out, or C_in for dinput).

    A 64-wide tile on a 32-channel layer leaves half its warps empty, and
    every thread still stages the half-zero tile; 32 halves the block instead.
    """
    return 32 if n <= 32 else _BLOCK_TILE_N


def threads_per_block(block_tile_m, block_tile_n):
    """One warp per 16x16 output fragment of the block tile."""
    return (block_tile_m // _WMMA_DIM) * (block_tile_n // _WMMA_DIM) * _WARP_SIZE

# RawModule options shared by every conv kernel. No -I for ROCm: hiprtc already
# searches its own install's include dir (where rocWMMA lives), and a -I from
# ROCM_PATH would be searched first, so a stale ROCM_PATH could shadow it.
_COMPILE_OPTIONS = ('-std=c++17',)


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
