import aether.config as config

_PHILOX_PREAMBLE = r'''
__device__ __forceinline__ float philox_uniform(
        unsigned long long philox_seed,
        unsigned long long philox_offset,
        long long idx) {

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
    return (c0 >> 8) * (1.0f / 16777216.0f);
}
'''

_philox_dropout_forward = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'T x, uint64 philox_seed, uint64 philox_offset, float64 keep_prob',
        'T y',
        '''
        float u = philox_uniform(philox_seed, philox_offset, i);
        y = (u < keep_prob) ? (T)(x / keep_prob) : (T)0;
        ''',
        '_philox_dropout_forward',
        preamble=_PHILOX_PREAMBLE,
    ),
    name='_philox_dropout_forward'
)

_philox_dropout_backward = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'T dvalues, uint64 philox_seed, uint64 philox_offset, float64 keep_prob',
        'T dinputs',
        '''
        float u = philox_uniform(philox_seed, philox_offset, i);
        dinputs = (u < keep_prob) ? (T)(dvalues / keep_prob) : (T)0;
        ''',
        '_philox_dropout_backward',
        preamble=_PHILOX_PREAMBLE,
    ),
    name='_philox_dropout_backward'
)

_philox_spatial_dropout_forward = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    'T x, uint64 philox_seed, uint64 philox_offset, float64 keep_prob, int64 C, int64 HWC',
    'T y',
    '''
    // Collapse the per-element index into a per-(sample, channel) index.
    long long c = (i / HWC) * C + i % C;
    float u = philox_uniform(philox_seed, philox_offset, c);
    y = (u < keep_prob) ? (T)(x / keep_prob) : (T)0;
    ''',
    '_philox_spatial_dropout_forward',
    preamble=_PHILOX_PREAMBLE,
    ),
    name = '_philox_spatial_dropout_forward'
)

_philox_spatial_dropout_backward = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    'T dvalues, uint64 philox_seed, uint64 philox_offset, float64 keep_prob, int64 C, int64 HWC',
    'T dinputs',
    '''
    // Compute the same channel mask used in forward pass, as the output is deterministic
    // passing in our forward inputs results in the same same u value.
    long long c = (i / HWC) * C + i % C;
    float u = philox_uniform(philox_seed, philox_offset, c);
    dinputs = (u < keep_prob) ? (T)(dvalues / keep_prob) : (T)0;
    ''',
    '_philox_spatial_dropout_backward',
    preamble=_PHILOX_PREAMBLE,
    ),
    name = '_philox_spatial_dropout_backward'
)

# Helper functions for layer Dropout
def is_gpu_dropout_available() -> bool:
    """Checks if CuPy hardware support and kernels are loaded."""
    return _philox_dropout_forward is not None and _philox_dropout_backward is not None

def philox_dropout_forward(inputs, seed: int, offset: int, keep_rate: float):
    return _philox_dropout_forward(inputs, seed, offset, float(keep_rate)) # type: ignore

def philox_dropout_backward(dvalues, seed: int, offset: int, keep_rate: float):
    return _philox_dropout_backward(dvalues, seed, offset, float(keep_rate)) # type: ignore

# Helper functions for layer SpatialDropout
def is_gpu_spatial_dropout_available() -> bool:
    """Checks if CuPy hardware support and kernels are loaded."""
    return _philox_spatial_dropout_forward is not None and _philox_spatial_dropout_backward is not None

def philox_spatial_dropout_forward(inputs, seed: int, offset: int, keep_rate: float, C: int, HWC: int):
    return _philox_spatial_dropout_forward(inputs, seed, offset, float(keep_rate), C, HWC) # type: ignore

def philox_spatial_dropout_backward(dvalues, seed: int, offset: int, keep_rate: float, C: int, HWC: int):
    return _philox_spatial_dropout_backward(dvalues, seed, offset, keep_rate, C, HWC) # type: ignore
