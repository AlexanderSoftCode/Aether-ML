"""Aether backend configuration, hardware dispatch, and execution runtime.

Central hardware abstraction layer (HAL) bridging CPU (NumPy) and GPU
(CuPy: CUDA / ROCm-HIP) execution paths. Imported by nearly every other
module in the framework.

Backend Resolution & Array Modules
-----------------------------------
xp                           -- active array module (numpy or cupy), mutated by set_backend()
HAS_CUPY / IS_HIP / IS_CUDA  -- detected hardware capability flags, resolved once at import
get_array_module(*arrays)    -- resolves the correct xp for a given tensor (cp.get_array_module or numpy fallback)
to_device(array, target)     -- explicit CPU<->GPU migration outside execution passes
set_backend(name)            -- switches the global xp / as_strided at runtime
enforce_cupy_env()           -- fail-fast guard for code paths that require GPU

Precision Policies
-------------------
DTypePolicy                  -- decouples compute dtype (float16/32/64) from float32 param storage
COMPUTE_DTYPE / PARAM_DTYPE  -- allowed compute dtypes / fixed master-weight dtype

Kernel Compilation & Dispatch
-------------------------------
build_kernel(factory, name)  -- deferred JIT builder for ElementwiseKernel/ReductionKernel/RawKernel objects
fuse_kernel(*args, **kwargs) -- decorator wrapping cp.fuse for elementwise kernels
resolve_gpu_launch_geometry()-- CUDA vs HIP variant + recommended threads-per-block (1024 / 512)
get_tensor_core_capable()    -- probes gfx9/11/12 (HIP) or compute capability >=70 (CUDA)

RNG & Training Synchronization
---------------------------------
TrainingClock                -- monotonic step counter shared across a model's stochastic layers
derive_stream_seed(seed, id) -- deterministic per-stream Philox seed via SeedSequence

Architectural Invariants
--------------------------
- Default execution degrades to CPU/NumPy if CuPy or a supported GPU is absent.
- Explicit GPU targets (enforce_cupy_env) fail fast rather than falling back silently.
- Layer weights and master parameters stay float32 unless explicitly cast for compute.
"""
import numpy as np
import warnings

# default backend incase user does not have CuPy package
xp = np
as_strided = np.lib.stride_tricks.as_strided
HAS_CUPY = False
IS_HIP = False
IS_CUDA = False
try:
    import cupy as cp
    if cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0:
        HAS_CUPY = True
        get_array_module = cp.get_array_module
        IS_HIP = bool(cp.cuda.runtime.is_hip)
        IS_CUDA = not IS_HIP
    else:
        HAS_CUPY = False
        cp = None
except Exception:
    HAS_CUPY = False
    cp = None

if not HAS_CUPY:
    def get_array_module(*args, **kwargs):
        return np
    
xp = cp if HAS_CUPY else np

# Used to have bfloat16, but currently isn't stable for this framework
COMPUTE_DTYPE = frozenset({'float16','float32', 'float64'})
PARAM_DTYPE = 'float32'


def enforce_cupy_env():
    """Loudly explodes only if a layer or user explicitly demands GPU execution."""
    if not HAS_CUPY:
        raise RuntimeError(
            "Failed to leverage backend logic for 'cupy'. "
            "Ensure that CuPy is installed and configured to your GPU environment.\n"
            "To install CuPy for AMD ROCm/NVIDIA CUDA, run: pip install cupy"
        )

def to_device(array, target='cupy'):
    """Explicity handles structural device migration outside execution passes."""

    if target.lower() == 'cupy':
        enforce_cupy_env()  # Explosive check triggers
        return cp.asarray(array)

    elif target.lower() == 'numpy':
        if HAS_CUPY and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)
    else:
        raise ValueError(
            f"Invalid target device '{target}'. Supported device backends are 'numpy' and 'cupy'"
        )
def get_stride_utility(xp_module):
    """Unified helper for nested stride pathways."""
    return xp_module.lib.stride_tricks.as_strided

def set_backend(backend_name):
    """Dynamically switches the global array namespace and layout views at runtime."""
    global xp, as_strided
    if backend_name.lower() == 'cupy':
        enforce_cupy_env()
        xp = cp
        as_strided = cp.lib.stride_tricks.as_strided
    elif backend_name.lower() == 'numpy':
        xp = np
        as_strided = np.lib.stride_tricks.as_strided
    else:
        raise ValueError(f"Unknown backend configuration context: {backend_name}")

def fuse_kernel(*fuse_args, **fuse_kwargs):
    """
    Decorator for cp.fuse'd elementwise kernels. Binds the name to a
    working fused callable when CuPy is available and custruction succeeds.
    """

    def decorator(func):
        if not HAS_CUPY:
            return None
        try:
            return cp.fuse(*fuse_args, **fuse_kwargs)(func)
        except Exception as e:
            warnings.warn(f"[aether] fuse_kernel: '{func.__name__}' failed to compile \n {e} ")
            return None
    return decorator

def build_kernel(factory, name=None):
    """
    Used for kernel objects that aren't a decorated function ex.ElementwiseKernel,
    cp.ReductionKernel, cp.RawKernel. 'factory' is a zero-arg callable that performs
    the actual construction, deferred so it's never evaluated while cp is None.
    
    EX: _leaky_relu_ew = config.build_kernel(lambda: config.cp.ElementwiseKernel(
    ...),
        '_leaky_relu_ew' 
    )
    """
    if not HAS_CUPY:
        return None
    try:
        return factory()
    except Exception as e:
        warnings.warn(f"[aether] build_kernel: '{name or getattr(factory, '__name__', '?')}' failed to compile: {e}")
        return None


def resolve_gpu_launch_geometry():
    """CUDA vs HIP variant + recommended total threads-per-block.
    512 threads is recommended for HIP, 1024 for CUDA -- see pooling_kernel.py adam_kernel.py
    or batch_norm_kernel.py for how each consumer maps this onto their own grid shape.
    """
    variant = "hip" if IS_HIP else "cuda"
    target_threads = 512 if IS_HIP else 1024
    return variant, target_threads

_TENSOR_CORE_CAPABLE = None

def get_tensor_core_capable():
    """Check if GPU supports tensor/matrix cores for optimized kernels.

    HIP: requires gfx9, gfx11, or gfx12 (RDNA and newer).
    CUDA: requires compute capability >= 70 (Volta and newer).

    Returns False if CuPy unavailable or probe fails.
    """
    global _TENSOR_CORE_CAPABLE
    if _TENSOR_CORE_CAPABLE is not None:
        return _TENSOR_CORE_CAPABLE

    if not HAS_CUPY:
        _TENSOR_CORE_CAPABLE = False
        return False

    try:
        device = cp.cuda.Device()
        if IS_HIP:
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            arch = props['gcnArchName']
            arch = arch.decode() if isinstance(arch, bytes) else arch
            _TENSOR_CORE_CAPABLE = (
                arch.startswith('gfx9') or
                arch.startswith('gfx11') or
                arch.startswith('gfx12')
            )
        else:
            cc = int(device.compute_capability)
            _TENSOR_CORE_CAPABLE = cc >= 70
    except Exception as e:
        warnings.warn(
            f"[aether] tensor core capability probe failed: {e}"
        )
        _TENSOR_CORE_CAPABLE = False

    return _TENSOR_CORE_CAPABLE

def derive_stream_seed(base_seed, stream_id):
    """
    Derives a deterministic 64-bit key for a specific random stream. The same
    `base_seed` and `stream_id` produce identical output across processes. A
    `base_seed` of None pulls fresh OS entropy, giving an unseeded stream.
    """
    if base_seed is None:
        entropy = None
        spawn_key = (int(stream_id),)
    else:
        entropy = [int(base_seed), int(stream_id)]
        spawn_key = ()

    seed_seq = np.random.SeedSequence(entropy, spawn_key=spawn_key)
    return int(seed_seq.generate_state(1, dtype=np.uint64)[0])
    
class TrainingClock:
    """
    Monotonic step counter shared by every stochastic layer in a model.

    Stochastic layers read `value` during forward but never mutate it; the training
    loop advances it exactly once per batch. This makes a repeated forward pass
    within the same step reproduce the same mask, and decouples backward from
    "whatever the stream head happens to be."
    """

    __slots__ = ("value",)

    def __init__(self, value=0):
        self.value = int(value)

    def advance(self):
        self.value += 1
        return self.value

    def reset(self, value=0):
        self.value = int(value)

class DTypePolicy():

    def __init__ (self, compute_dtype: str | None = None) -> None:

        if compute_dtype is not None and not isinstance(compute_dtype, str):
            raise TypeError(
                f"compute_dtype must be a str or None, but got {type(compute_dtype).__name__!r}."
            )
                            
        if compute_dtype is not None and compute_dtype not in COMPUTE_DTYPE:
            raise ValueError(f"dtype {compute_dtype} not in {COMPUTE_DTYPE}")

        if compute_dtype == 'bfloat16' and xp is np:
            raise RuntimeError(f"dtype {compute_dtype} is not supported in NumPy. Please use another value")

        if compute_dtype == 'float16' and xp is np:
            warnings.warn(
                "NumPy float16 is emulated, consider float32 for CPU work.",
                UserWarning,
                stacklevel=2,
            )

        self.compute_dtype_name = compute_dtype
        self.param_dtype = np.dtype(PARAM_DTYPE)

    def cast_to_compute(self, *tensors):
        """
        Casts incoming tensors to compute precision.
        Returns a single tensor if one argument is passed, or a tuple if multiple arguments are passed.
        """
        if not tensors:
            return ()

        if self.compute_dtype_name is None:
            return tensors[0] if len(tensors) == 1 else tensors

        casted = [
            t.astype(get_array_module(t).dtype(self.compute_dtype_name), copy=False)
            if t is not None else None
            for t in tensors
        ]

        return casted[0] if len(casted) == 1 else tuple(casted)

    def cast_to_param(self, *tensors):
        """
        Casts gradients/tensors back to master parameter precision.
        Returns a single tensor if one argument is passed, or a tuple if multiple arguments are passed.
        """
        if not tensors:
            return ()

        casted = [
            t.astype(get_array_module(t).dtype(self.param_dtype), copy=False)
            if t is not None else None
            for t in tensors
        ]

        return casted[0] if len(casted) == 1 else tuple(casted)