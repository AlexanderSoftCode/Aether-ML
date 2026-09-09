# aether/preprocessing/_utils.py
import numpy as np
import warnings
import aether.config as config

TENSOR_DTYPES = {'float16', 'bfloat16', 'float32', 'float64'}  # valid dtypes for ToTensor/to_tensor

# The four helper functions below are used by 
# aether.preprocessing.transforms.to_tensor
def validate_dtype(dtype):
    if dtype is None:
        return None
    try:
        dtype_name = np.dtype(dtype).name
    except (TypeError, ValueError):
        dtype_name = str(dtype).split('.')[-1]

    if dtype_name not in TENSOR_DTYPES:
        raise ValueError(f"Invalid dtype '{dtype}'. Expected one of: {sorted(TENSOR_DTYPES)}")
    if dtype_name == "float64":
        warnings.warn("float64 precision can significantly decrease speed...", UserWarning, stacklevel=3)
    return dtype

def is_dtype_like(val) -> bool:
    if val is None:
        return True
    if isinstance(val, (tuple, list)):
        return all(is_dtype_like(item) for item in val)
    if isinstance(val, (type, np.dtype, str)):
        try:
            np.dtype(val)
            return True
        except (TypeError, ValueError):
            return str(val).split('.')[-1] in TENSOR_DTYPES
    return False

def parse_inputs(args, kw_dtype):
    args = list(args)
    if not args:
        return [], None
    if kw_dtype is None and is_dtype_like(args[-1]):
        kw_dtype = args.pop()
    return args, kw_dtype

def resolve_dtypes(raw_dtype, num_arrays):
    if raw_dtype is None:
        return [None] * num_arrays
    
    is_tuple = isinstance(raw_dtype, (tuple, list))
    dtypes = tuple(validate_dtype(d) for d in raw_dtype) if is_tuple else (validate_dtype(raw_dtype),)

    if len(dtypes) > 1 and len(dtypes) != num_arrays:
        raise ValueError(f"Length of `dtype` tuple ({len(dtypes)}) must match number of arrays ({num_arrays}).")

    resolved = list(dtypes) if len(dtypes) > 1 else [dtypes[0]] * num_arrays
    return resolved


def convert_single_tensor(arr, target_dtype, target_device, preserve_integers):
    if arr is None:
        return None
    try:
        tensor = config.to_device(arr, target=target_device)
    except RuntimeError:
        tensor = np.asarray(arr)

    if target_dtype is None or not hasattr(tensor, 'astype'):
        return tensor

    if preserve_integers and target_dtype is None and tensor.dtype.kind in ('i', 'u'):
        return tensor

    return tensor.astype(target_dtype, copy=False)