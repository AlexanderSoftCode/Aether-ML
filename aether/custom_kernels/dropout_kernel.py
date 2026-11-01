import warnings
from string import Template

import numpy as np

import aether.config as config

_PHILOX_PREAMBLE = r'''
__device__ __forceinline__ void philox4x32_10(
        unsigned long long philox_seed,
        unsigned long long philox_offset,
        long long idx,
        unsigned int* words) {

    unsigned int c0 = (unsigned int)(philox_offset & 0xffffffffULL);
    unsigned int c1 = (unsigned int)(philox_offset >> 32);
    unsigned int c2 = (unsigned int)((unsigned long long)idx & 0xffffffffULL);
    unsigned int c3 = (unsigned int)((unsigned long long)idx >> 32);

    unsigned int k0 = (unsigned int)(philox_seed & 0xffffffffULL);
    unsigned int k1 = (unsigned int)(philox_seed >> 32);

    #pragma unroll
    for (int round = 0; round < 10; round++) {
        unsigned long long p0 = (unsigned long long)0xD2511F53u * c0;
        unsigned long long p1 = (unsigned long long)0xCD9E8D57u * c2;
        unsigned int lo0 = (unsigned int)(p0 & 0xffffffffULL);
        unsigned int hi0 = (unsigned int)(p0 >> 32);
        unsigned int lo1 = (unsigned int)(p1 & 0xffffffffULL);
        unsigned int hi1 = (unsigned int)(p1 >> 32);

        unsigned int nc0 = hi1 ^ c1 ^ k0;
        unsigned int nc1 = lo1;
        unsigned int nc2 = hi0 ^ c3 ^ k1;
        unsigned int nc3 = lo0;

        c0 = nc0; c1 = nc1; c2 = nc2; c3 = nc3;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    words[0] = c0; words[1] = c1; words[2] = c2; words[3] = c3;
}
'''

# Each thread owns the 8 elements starting at thread_index * 8 and splits one Philox
# call into their 16-bit draws, low half first. threshold is compared as 32 bits so
# 65536 (rate 0) keeps every draw. Forward and backward launch this same source, and
# the draws never see the data, so both directions drop the same elements in any dtype.
_DROPOUT_TEMPLATE = Template(r'''
$include
$philox

extern "C" __global__
void $kernel_name(
    const $data_t* __restrict__ src,
    $data_t* __restrict__ dst,
    const unsigned long long philox_seed,
    const unsigned long long philox_offset,
    const unsigned int threshold,
    const $work_t inv_keep,
    const long long size
) {
    const long long thread_index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long first = thread_index * 8;
    if (first >= size) return;

    unsigned int words[4];
    philox4x32_10(philox_seed, philox_offset, thread_index, words);

    for (int j = 0; j < 8 && first + j < size; ++j) {
        const unsigned int draw = (words[j >> 1] >> ((j & 1) * 16)) & 0xffffu;
        const long long i = first + j;
        dst[i] = $store(draw < threshold ? $load(src[i]) * inv_keep : ($work_t)0);
    }
}
''')

_INCLUDES = {
    "hip": "#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>",
    "cuda": "#include <cuda_fp16.h>",
}

_DTYPES = {
    np.dtype(np.float16): dict(data_t='half', work_t='float', load='__half2float', store='__float2half'),
    np.dtype(np.float32): dict(data_t='float', work_t='float', load='', store=''),
    np.dtype(np.float64): dict(data_t='double', work_t='double', load='', store=''),
}

# A dropped channel is exactly 0: multiplying alone would turn NaN or inf into NaN.
_channel_mask_apply = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'T x, U mask',
        'T y',
        'y = mask != 0 ? (T)(x * mask) : (T)0',
        'channel_mask_apply_kernel',
    ),
    name='_channel_mask_apply',
)

_dropout_kernel_cache = {}


def _get_compiled_kernel(variant: str, dtype):
    """Cached, compiled kernel for one vendor variant and data dtype.

    A failed compile is cached as None so a broken build doesn't re-attempt (and
    re-warn) on every launch.
    """
    key = (variant, dtype)
    if key in _dropout_kernel_cache:
        return _dropout_kernel_cache[key]

    if dtype not in _DTYPES:
        raise TypeError(f"[aether] GPU dropout reads float16, float32 or float64, not {dtype}.")

    kernel_name = f"philox_dropout_{dtype.name}_{variant}_kernel"
    source = _DROPOUT_TEMPLATE.substitute(
        include=_INCLUDES[variant],
        philox=_PHILOX_PREAMBLE,
        kernel_name=kernel_name,
        **_DTYPES[dtype],
    )

    try:
        kernel = config.cp.RawModule(code=source).get_function(kernel_name)
    except (config.cp.cuda.compiler.CompileException, config.cp.cuda.driver.CUDADriverError) as e:
        warnings.warn(f"[aether] dropout_kernel: {kernel_name} compile failed: {e}")
        kernel = None

    _dropout_kernel_cache[key] = kernel
    return kernel


def is_gpu_dropout_available() -> bool:
    """Checks if CuPy hardware support is loaded. Kernels compile at first launch."""
    return config.HAS_CUPY


def apply_channel_mask(x, mask):
    return _channel_mask_apply(x, mask)


def philox_dropout(src, seed: int, offset: int, threshold: int, inv_keep: float):
    """Returns src with each element kept and scaled by inv_keep, or zeroed.

    The result has src's shape and dtype. Returns None when the kernel for that
    dtype failed to compile, so the caller can fall back.
    """
    variant, threads = config.resolve_gpu_launch_geometry()
    kernel = _get_compiled_kernel(variant, np.dtype(src.dtype))
    if kernel is None:
        return None

    xp = config.cp
    src = xp.ascontiguousarray(src)
    dst = xp.empty(src.shape, dtype=src.dtype)
    size = src.size
    philox_calls = -(-size // 8)
    blocks = -(-philox_calls // threads)
    scale = np.float64(inv_keep) if src.dtype == xp.float64 else np.float32(inv_keep)

    kernel(
        (blocks,), (threads,),
        (src, dst, np.uint64(seed), np.uint64(offset), np.uint32(threshold), scale, np.int64(size)),
    )
    return dst
