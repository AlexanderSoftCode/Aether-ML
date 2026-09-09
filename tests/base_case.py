import warnings
import unittest
import numpy as np
from aether.config import set_backend
import aether.config as config

try:
    import safetensors.numpy as safetensors_np
    HAS_SAFETENSORS = True
except ImportError:
    safetensors_np = None
    HAS_SAFETENSORS = False

BACKENDS_TO_TEST = ['numpy']
if config.HAS_CUPY:
    BACKENDS_TO_TEST.append('cupy')

if config.HAS_CUPY:
    import cupy as cp


class AetherBaseTestCase(unittest.TestCase):
    """
    A foundational test case structure providing global array module
    pointers and standard environment cleaning methods between hardware
    backend swaps.
    """
    __test__ = False  # Suppress standalone discovery of the base class (pytest only, see note above)
    backend_name = "numpy"  # Fallback default backend

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        # Suppress test runners from registering intermediate base classes
        cls.__test__ = not ("Base" in cls.__name__)

    def setUp(self):
        super().setUp()

        # Reads self.backend_name set by class definition
        config.set_backend(backend_name=self.backend_name)
        self.xp = config.xp

        # CuPy's own np.pad() shim trips NumPy 2.5's shape-assignment deprecation
        # internally (cupy/_padding/pad.py) whenever a padded Conv/Pool layer runs
        # on the cupy backend.
        self._pad_warn_ctx = warnings.catch_warnings()
        self._pad_warn_ctx.__enter__()
        warnings.filterwarnings(
            "ignore",
            message=r".*Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )

    def shortDescription(self):
        """Docstrings are ommitted when running verbose -v unit tests"""
        return None

    def make_built_layer(self, layer_cls, input_shape: tuple[int, ...], seed: int | None = None, **kwargs):
        """Constructs, binds to the active device backend, and builds a layer instance.

        This test helper standardizes the parameterized layer lifecycle by first
        instantiating the layer, triggering device compilation/runtime pointer
        rebinding if supported, and finally executing the layer's build routine to
        allocate parameter buffers and compute output spatial shapes.

        Args:
            layer_cls: The layer class to instantiate (e.g., `Conv`, `Dense`).
            input_shape: Feature/spatial shape tuple EXCLUDING the batch dimension
                (e.g., `(28, 28, 1)` for 2D Conv or `(128,)` for Dense).
            seed: Optional integer seed for deterministic parameter initialization.
                Defaults to None.
            **kwargs: Arbitrary keyword arguments forwarded directly to the
                `layer_cls` constructor (e.g., `out_channels`, `stride`, `padding`).

        Returns:
            Layer: A fully initialized, device-compiled, and built layer instance.

        Example:
            >>> class TestConvLayer(AetherBaseLayerTestCase):
            ...     def setUp(self):
            ...         super().setUp()
            ...         self.layer = self.make_built_layer(
            ...             Conv,
            ...             input_shape=(28, 28, 1),
            ...             seed=42,
            ...             in_channels=1,
            ...             out_channels=16,
            ...             filter_size=(3, 3),
            ...             stride=(1, 1),
            ...             padding="same",
            ...         )
        """
        layer = layer_cls(**kwargs)

        if hasattr(layer, "_compile_for_device"):
            layer._compile_for_device(self.backend_name)

        layer.build(input_shape, seed=seed)
        return layer

    def make_component(self, component_cls, **kwargs):
        """Instantiates any non-layer aether-component (e.g. Loss, Optimizer, etc.)
        and compiles backend_specific-kernels if supported.
        """
        instance = component_cls(**kwargs)
        if hasattr(instance, "_compile_for_device"):
            instance._compile_for_device(self.backend_name)
        return instance

    def tearDown(self):
        """Reset tracking state to system NumPy default safely between tests."""
        if hasattr(self, "_pad_warn_ctx"):
            self._pad_warn_ctx.__exit__(None, None, None)
        if config.HAS_CUPY:
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        set_backend(backend_name='numpy')

class AetherBaseLayerTestCase(AetherBaseTestCase):
    """Base class for layers/activations/loss modules"""
    __test__ = False

    def set_precision(self, layer, compute_dtype):
        """Helper mirroring Model.set_precision behavior for an individual layer."""
        # Since I find them annoying, we'll ignore the known emulation warnings after
        # test setup
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*NumPy float16 is emulated.*",
                category=UserWarning
            )
            policy = config.DTypePolicy(compute_dtype=compute_dtype)
            if hasattr(layer, "_apply_precision") and not getattr(
                layer, "_precision_exempt", False
            ):
                layer._apply_precision(policy)
        return policy

def register_test_suites(target_globals, template_cls):
    """
    Generates backend subclasses in globals and removes the base template
    (and any of its discoverable TestCase ancestors) so test runners only
    see and execute the backend-specific suites.
    """
    for ancestor in template_cls.__mro__:
        if (
            issubclass(ancestor, unittest.TestCase)
            and ancestor.__dict__.get("__test__", None) is False
            and target_globals.get(ancestor.__name__) is ancestor
        ):
            target_globals.pop(ancestor.__name__, None)

    target_globals.pop(template_cls.__name__, None)

    for backend in BACKENDS_TO_TEST:
        class_name = f"{template_cls.__name__}_{backend.upper()}"
        subclass = type(
            class_name,
            (template_cls,),
            {
                "backend_name": backend,
                "__test__": True,
                "__module__": template_cls.__module__,
            },
        )
        target_globals[class_name] = subclass