import warnings
import numpy as np
from typing import Sequence, Union, Tuple, Optional, Any
import aether.config as config
from ._utils import parse_inputs, resolve_dtypes, convert_single_tensor

TENSOR_DTYPES = {'float16', 'bfloat16', 'float32', 'float64'}  # valid dtypes for ToTensor/to_tensor
STATS_DTYPES = {'float32', 'float64'}  # valid dtypes for statistics-based transforms (StandardScaler)

def _dtype_name(dt):
    """JSON-safe stringification of a dtype spec (str/type/np.dtype/sequence)."""
    if dt is None or isinstance(dt, str):
        return dt
    if isinstance(dt, (tuple, list)):
        return [_dtype_name(d) for d in dt]
    return np.dtype(dt).name

def to_tensor(
    *args: Any,
    target_device: Optional[str] = None,
    dtype: Union[str, type, np.dtype, Sequence[Union[str, type, np.dtype]], None] = None,
    preserve_integers: bool = True
) -> Union[Any, Tuple[Any, ...], None]:
    """Converts array-like inputs into backend tensors (NumPy or CuPy).

    Flexible API for precision handling and hardware device placement. Accepts
    individual arrays or multiple positional arrays (e.g., training/testing sets,
    feature/label splits) and casts them based on provided specifications.

    Args:
        *args: One or more array-like objects to convert. May optionally include
            a trailing positional argument specifying target `dtype`(s).
        target_device: Hardware execution backend ('numpy' or 'cupy'). If None,
            defaults to the active device specified in `aether.config`.
        dtype: Desired floating-point data type. Can be passed as a single dtype
            (broadcasted across all arrays) or a sequence/tuple matching the number
            of passed arrays 1:1.
        preserve_integers: If True, bypasses `dtype` casting for integer-based arrays
            (e.g., target labels or indices) to prevent array indexing bugs.

    Returns:
        Converted backend tensor if a single input array was provided, a tuple of
        converted backend tensors if multiple inputs were provided, or None if input
        is empty.

    Raises:
        ValueError: If `dtype` tuple length does not match the number of input arrays,
            or if an unsupported `dtype` string/type is provided.

    Warns:
        UserWarning: If `float64` precision is specified, cautioning against hardware
            performance degradation.

    Examples:
        >>> import aether as ae
        >>> X_tr, y_tr = ae.to_tensor(X_tr, y_tr, dtype='float32')
        >>> X_tr, X_ts, y_tr, y_ts = ae.to_tensor(X_tr, X_ts, y_tr, y_ts, dtype=('float32', 'float32', None, None))
    """
    arrays, raw_dtype = parse_inputs(args, dtype)
    if not arrays:
        return None

    if target_device is None:
        target_device = 'cupy' if (config.HAS_CUPY and config.xp == config.cp) else 'numpy'

    target_dtypes = resolve_dtypes(raw_dtype, len(arrays))
    is_sequence = isinstance(raw_dtype, (tuple,list))

    results = []
    for i, (arr, dt) in enumerate(zip(arrays, target_dtypes)):
        if arr is None:
            results.append(None)
            continue
        try:
            temp_tensor = config.to_device(arr, target=target_device)
        except RuntimeError:
            temp_tensor = np.asarray(arr)

        if (
            len(arrays) > 1
            and i > 0
            and preserve_integers
            and not is_sequence
            and hasattr(temp_tensor, 'dtype')
            and temp_tensor.dtype.kind in ('i','u')
        ):
            effective_dtype = None
        else:
            effective_dtype = dt

        results.append(convert_single_tensor(temp_tensor, effective_dtype, target_device, preserve_integers))
    return results[0] if len(results) == 1 else tuple(results)


class Preprocess:
    """Base class for all preprocessing/transform components.
    
    This base class is handled by Model.configure, and Model.finalize. The subclasses will
    implement 'transform' and 'fit'.

    Attributes:
        _precision_exempt (bool): Opt-out of model-driven precision dispatch"""
    is_fitted = True
    _precision_exempt: bool = False

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, cfg):
        """Reconstructs an instance from a mapping produced by `get_config()`.

        The default assumes a flat, constructor-shaped config. Transforms whose
        config is not directly constructor-shaped (a container holding nested
        entries, or one carrying derived state that is not a constructor
        argument) override this.

        Args:
            cfg (dict): The mapping produced by `get_config()`.

        Returns:
            Preprocess: A newly constructed instance of `cls`.
        """
        return cls(**cfg)
 
    def fit(self, *args, **kwargs):
        return self
 
    def transform(self, *args, **kwargs):
        raise NotImplementedError
 
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
 
    def _compile_for_device(self, device):
        pass
 
    def _apply_precision(self, policy):
        pass

class ToTensor(Preprocess):
    """Callable pipeline transform wrapper for converting inputs into backend tensors.

    Stores configuration parameters for precision and device targets so that
    tensor conversion can be consistently executed as part of an end-to-end data pipeline.

    Args:
        dtype: Desired target precision for floating-point tensors ('float16',
            'float32', etc.). Can be a single dtype or a tuple matching positional input arguments.
        preserve_integers: If True, integer-typed arrays (e.g., targets or labels)
            retain their integer precision during evaluation. Defaults to True.
        target_device: Specific execution backend target ('numpy' or 'cupy'). If None,
            defers to active environment setup in `aether.config`.
    """
    def __init__(self, dtype=None, preserve_integers=True, target_device=None):
        self.dtype = dtype
        self.preserve_integers = preserve_integers
        self.target_device = target_device
        self._device_pinned = target_device is not None

    def transform(self, *arrays):
        """Executes tensor conversion on provided arrays using pre-configured settings.

        Args:
            *arrays: One or more array-like objects to transform.

        Returns:
            Single converted array or tuple of converted arrays.
        """
        return to_tensor(
            *arrays,
            target_device=self.target_device,
            dtype=self.dtype,
            preserve_integers=self.preserve_integers
        )

    def _compile_for_device(self, device):
        """Model.to() owns the device target -- it always wins, but a user who
        explicitly pinned target_device is warned that their pin is being discarded
        rather than having it silently overridden."""
        if self._device_pinned and self.target_device != device:
            warnings.warn(
                f"[aether] ToTensor(target_device='{self.target_device}') was explicitly "
                f"set, but model.to('{device}') overrides it -- a preprocessing "
                "pipeline's device always follows the model. The pinned value is discarded.",
                UserWarning,
                stacklevel=2,
            )
        self.target_device = device

    def _apply_precision(self, policy):
        """Model.set_precision() only fills the cast dtype if the user has not pinned one."""
        if self.dtype is None and policy.compute_dtype_name is not None:
            self.dtype = policy.compute_dtype_name

    def get_config(self):
        return {
            "dtype": _dtype_name(self.dtype),
            "preserve_integers": self.preserve_integers,
            "target_device": self.target_device,
        }

class StandardScaler(Preprocess):
    """
    Calculates mean and standard deviation per feature during initialization/fitting,
    and applies standardization centering to input datasets. 

    Args:
        mean: Optional pre-computed mean vector.
        std: Optional pre-computed standard deviation vector.
        axis: Axis or axes along which statistics are computed
            -None: Scaler normalaztion across all axes
            - 0: Feature-wise normalaztion across 2D tabular data (S, D).
            - (0, 1, 2): Channel-wise normalization for SHWC images (S, H, W, C):
        dtype: Precision for the computed/stored statistics. Must be 'float32'
            (default) or 'float64' -- mean/std are never computed or stored below
            single precision, regardless of the input array's own dtype.

    Raises:
        ValueError: If `dtype` is given and is not 'float32' or 'float64'.

    Note:
        Exempt from model-driven precision dispatch due to half-precision overflow
    """
    _precision_exempt = True

    def __init__(self, mean=None, std=None, axis = None, dtype=None):
        if dtype is not None:
            dtype_name = np.dtype(dtype).name
            if dtype_name not in STATS_DTYPES:
                raise ValueError(
                    f"[aether] StandardScaler dtype must be one of {sorted(STATS_DTYPES)} -- "
                    "mean/std statistics are never computed below single (float32) "
                    f"precision. Got '{dtype_name}'."
                )
        self.dtype = np.dtype(dtype) if isinstance(dtype, str) else dtype
        self.axis = tuple(axis) if isinstance(axis, list) else axis

        self._dtype_pinned = dtype is not None

        self.mean = self._coerce(mean)
        self.std = self._coerce(std)

    def _coerce(self, v):
        if v is None:
            return v
        target = self.dtype if self.dtype is not None else np.float32
        if hasattr(v, "dtype"):
            # Below the fp32 floor (or non-float): promote. Already fp32/fp64: keep as-is.
            return v if np.dtype(v.dtype).name in STATS_DTYPES else v.astype(target, copy=False)
        return np.asarray(v, dtype=target)

    @property
    def is_fitted(self):
        return self.mean is not None

    def fit(self, X):
        """Computes feature mean and standard deviation from dataset X.

        Args:
            X: Input dataset array used to compute normalization statistics.

        Returns:
            The fitted instance of StandardScaler.
        """
        xp = config.get_array_module(X)

        if self.dtype is None:
            # Integers and anything below the fp32 floor (float16/bfloat16) promote to
            # float32; an incoming float32/float64 array keeps its own precision.
            if np.issubdtype(X.dtype, np.integer) or np.dtype(X.dtype).name not in STATS_DTYPES:
                self.dtype = np.dtype(np.float32)
            else:
                self.dtype = X.dtype

        accum_dtype = np.promote_types(self.dtype, np.float32)
        X_float = X.astype(accum_dtype, copy=False)
        self.mean = xp.mean(X_float, axis=self.axis, keepdims=True).astype(self.dtype, copy=False)
        self.std = xp.std(X_float, axis=self.axis, keepdims=True).astype(self.dtype, copy=False)
        return self

    def transform(self, X):
        """Applies feature scaling to input matrix X.

        Args:
            X: Input array to scale using computed mean and standard deviation.

        Returns:
            Standardized feature tensor.

        Raises:
            ValueError: If called before `fit()` has been run or without passing `mean`/`std`
                to `__init__`.
        """
        clip = 1e-8
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fit or provided mean/std before calling")
        xp = config.get_array_module(X)

        if config.get_array_module(self.mean) != xp:
            self.mean = xp.asarray(self.mean)
            self.std = xp.asarray(self.std)
        if self.dtype is not None and hasattr(X, "astype") and X.dtype != self.dtype:
            X = X.astype(self.dtype, copy=False)
        return (X-self.mean) / (self.std + clip)

    def _compile_for_device(self, device):
        """Stat migration on Model.to()."""
        if self.mean is None:
            return
        self.mean = config.to_device(self.mean, target=device)
        self.std = config.to_device(self.std, target=device)

    def _apply_precision(self, policy):
        """Model.set_precision() owns the cast dtype unless the constructor pinned one."""
        if self._dtype_pinned or policy.compute_dtype_name is None:
            return

        new_dtype = np.dtype(policy.compute_dtype_name)
        if self.dtype is not None and new_dtype == self.dtype:
            return
        self.dtype = new_dtype

        if self.mean is not None:
            self.mean = self.mean.astype(new_dtype, copy=False)
        if self.std is not None:
            self.std = self.std.astype(new_dtype, copy=False)

    def get_config(self):
        mean_host = config.to_device(self.mean, target="numpy") if self.mean is not None else None
        std_host = config.to_device(self.std, target="numpy") if self.std is not None else None
        return {
            "mean": mean_host.tolist() if mean_host is not None else None,
            "std": std_host.tolist() if std_host is not None else None,
            "axis": list(self.axis) if isinstance(self.axis, tuple) else self.axis,
            "dtype": _dtype_name(self.dtype),
            "dtype_pinned": bool(self._dtype_pinned),
        }

    @classmethod
    def from_config(cls, cfg):
        """Rebuilds a scaler, restoring the dtype pin exactly as it was at save time.

        `get_config()` always emits `dtype`, including one `fit()` inferred, so a
        plain `cls(**cfg)` would come back pinned and silently ignore any later
        `Model.set_precision()`. The recorded pin is therefore applied after
        construction rather than through `__init__`, which keeps derived state out
        of the public constructor signature.

        Args:
            cfg (dict): The mapping produced by `get_config()`.

        Returns:
            StandardScaler: The restored scaler, pinned only if the original was.
        """
        cfg = dict(cfg)
        dtype_pinned = bool(cfg.pop("dtype_pinned", False))
        scaler = cls(**cfg)
        scaler._dtype_pinned = dtype_pinned
        return scaler

class Rescale(Preprocess):
    """Scales array inputs by a multiplier factor (e.g., 1/255.0 for sRGB uint8 images)."""
    def __init__(self, factor: float = 1.0 / 255.0):
        self.factor = factor

    def get_config(self):
        return {"factor": self.factor}
 
    def _apply_factor(self, X):
        if hasattr(X, 'dtype') and X.dtype.kind in ('u', 'i'):
            return X * np.float32(self.factor)
        return X * self.factor
 
    def transform(self, *args):
        if not args:
            return None
 
        if len(args) == 1:
            return self._apply_factor(args[0])
 
        # If multiple inputs (e.g., X, y) are passed through Compose,
        # only scale floating-point feature tensors and leave integer labels untouched.
        results = []
        for arr in args:
            if hasattr(arr, 'dtype') and arr.dtype.kind in ('f',):
                results.append(self._apply_factor(arr))
            else:
                results.append(arr)
        return tuple(results)

    
class Compose(Preprocess):
    """Sequentially chains multiple data transformations into an end-to-end preprocessing pipeline.

    Executes a list of transformations in sequential order, passing the output
    of each transform as the input to the next. Supports both stateless transforms
    (e.g., `ToTensor`, `Rescale`) and stateful transforms (e.g., `StandardScaler`).

    Args:
        transforms (Sequence[Preprocess]): An ordered sequence of `Preprocess`
            instances to execute. May be empty, which yields an identity pipeline.

    Attributes:
        transforms (list[Preprocess]): The list of chained transforms.

    Raises:
        TypeError: If any member of `transforms` is not a `Preprocess` instance.

    Examples:
        >>> import aether as ae
        >>> import numpy as np
        >>>
        >>> # Prepare raw input arrays (e.g., uint8 image batches)
        >>> X_train = np.random.randint(0, 256, size=(50000, 32, 32, 3), dtype=np.uint8)
        >>> X_test = np.random.randint(0, 256, size=(10000, 32, 32, 3), dtype=np.uint8)
        >>>
        >>> # Build and fit pipeline on training data 
        >>> TARGET_DEVICE = "cupy"
        >>> feature_pipeline = ae.Compose([
        ...     ae.ToTensor(dtype='float32', target_device=TARGET_DEVICE),
        ...     ae.Rescale(factor=1.0 / 255.0),
        ...     ae.StandardScaler()
        ... ]).fit(X_train)
        >>>
        >>> # Transform training and test partitions using fitted statistics
        >>> X_train_tensor = feature_pipeline(X_train)
        >>> X_test_tensor = feature_pipeline(X_test)
        >>>
        >>> # Alternatively, use fit_transform for a single-pass workflow
        >>> X_train_tensor = feature_pipeline.fit_transform(X_train)
    """
    def __init__(self, transforms: Sequence[Any]):
        self.transforms = list(transforms)
        for i, transform in enumerate(self.transforms):
            if not isinstance(transform, Preprocess):
                raise TypeError(
                    f"Expected an instance of 'Preprocess' at transforms[{i}], "
                    f"but got '{type(transform).__name__}'. Make sure every transform "
                    "you pass in inherits from aether.preprocessing.Preprocess."
                )

    @property
    def is_fitted(self):
        return all(getattr(t, "is_fitted", True) for t in self.transforms)

    def _compile_for_device(self, device):
        for t in self.transforms:
            t._compile_for_device(device)

    def _apply_precision(self, policy):
        """Mirrors Model.set_precision()'s layer loop"""
        for t in self.transforms:
            if not getattr(t, "_precision_exempt", False):
                t._apply_precision(policy)

    def fit_transform(self, *args: Any):
        """Fits stateful transforms and returns the final transformed result
        in a single pass.
        """
        res = args[0] if len(args) == 1 else args

        for transform in self.transforms:
            # Check for a dedicated fit_transform method on the individual transform
            if hasattr(transform, "fit_transform") and callable(transform.fit_transform):
                res = transform.fit_transform(*res) if isinstance(res, tuple) else transform.fit_transform(res)
            else:
                # Otherwise fit if stateful, then transform
                if hasattr(transform, "fit") and callable(transform.fit):
                    if isinstance(res, tuple):
                        transform.fit(*res)
                    else:
                        transform.fit(res)

                if callable(transform):
                    res = transform(*res) if isinstance(res, tuple) else transform(res)

        return res

    def fit(self, *args: Any):
        """Fits all stateful transforms in the pipeline sequentially and returns self."""
        self.fit_transform(*args)
        return self
    
    def transform(self, *args):
        """Runs every chained transform in order, feeding each output to the next.

        Args:
            *args: One or more array-like objects to push through the pipeline.

        Returns:
            The output of the final transform: a single array, or a tuple if the
            last transform emitted multiple arrays. Returns the input unchanged
            for an empty pipeline.
        """
        res = args[0] if len(args) == 1 else args
        for transform in self.transforms:
            if isinstance(res, tuple):
                res = transform(*res)
            else:
                res = transform(res)
        return res

    def get_config(self):
        return {
            "transforms": [
                {"class_name": type(t).__name__, "config": t.get_config()}
                for t in self.transforms
            ]
        }

    @classmethod
    def from_config(cls, cfg):
        """Rebuilds the chain, resolving every member entry through `deserialize`.

        Recursive by construction: a nested `Compose` entry re-enters here through
        `deserialize`. Members are handed to `__init__` rather than assigned, so its
        `Preprocess` check rejects anything a manifest resolved to something else.

        Args:
            cfg (dict): A `{"transforms": [{class_name, config}, ...]}` mapping.

        Returns:
            Compose: A pipeline holding the reconstructed members, in saved order.

        Raises:
            ValueError: If a member entry names a class this module does not define.
            TypeError: If a member entry does not resolve to a concrete `Preprocess`.
        """
        return cls(transforms=[deserialize(entry) for entry in cfg.get("transforms") or []])


def deserialize(entry: dict | None) -> Optional[Preprocess]:
    """Reconstructs a Preprocess transform or pipeline from a serialized configuration dictionary.

    This function acts as the universal factory and resolver for preprocessing 
    transforms. It dynamically resolves the target transform class, validates it, 
    and reconstructs the instance (including nested pipelines like `Compose` and 
    stateful objects like `StandardScaler`) using the class's `from_config` method.

    When is this used?
    ------------------
    - **Internally:** Called automatically by `Model.load()` to restore attached 
      data preprocessing pipelines from saved `.aether` model archives.
    - **Standalone:** Called by users to load or transfer preprocessor pipelines 
      independently of a model (e.g., loading fitted scalers into external APIs, 
      ETL data workers, or configuration-driven workflows).

    Args:
        entry (dict | None): A dictionary containing `{"class_name": str, "config": dict}` 
            as produced by a transform's `get_config()` method (or `Model.save()`). 
            Passing `None` safely returns `None`.

    Returns:
        Preprocess | None: An instantiated and configured `Preprocess` transform 
        (e.g., `Compose`, `ToTensor`, `StandardScaler`), or `None` if `entry` is empty/None.

    Raises:
        ValueError: If `class_name` is not a known preprocessor class in this module.
        TypeError: If the resolved class does not inherit from `Preprocess`, or if 
            the parameters inside `config` do not match the expected constructor signature.

    Example:
        Reconstructing an entire multi-step pipeline:

        >>> pipeline = ae.Compose([ae.ToTensor(dtype="float32"), ae.Rescale(1.0 / 255.0)])
        >>> entry = {"class_name": "Compose", "config": pipeline.get_config()}
        >>> restored_pipeline = ae.preprocessing.deserialize(entry)
    """
    if not entry:
        return None

    class_name = entry["class_name"]
    cfg = entry.get("config") or {}

    transform_cls = globals().get(class_name)

    # Unknown class check
    if transform_cls is None:
        raise ValueError(
            f"[aether] Unknown preprocessor class '{class_name}' found in saved manifest. "
            f"Only classes defined in '{__name__}' can be deserialized."
        )

    # Base class rejection
    if transform_cls is Preprocess or not (
        isinstance(transform_cls, type) and issubclass(transform_cls, Preprocess)
    ):
        raise TypeError(
            f"[aether] Manifest entry '{class_name}' is not a valid concrete "
            f"'Preprocess' subclass."
        )

    try:
        return transform_cls.from_config(cfg)
    except TypeError as exc:
        raise TypeError(
            f"[aether] Could not reconstruct preprocessor '{class_name}' from saved config "
            f"{cfg}. The class signature may have changed since the model was saved."
        ) from exc