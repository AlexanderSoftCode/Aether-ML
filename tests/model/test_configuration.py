import warnings

import numpy as np

from aether.layers.linear import Dense
from aether.losses import SoftmaxCategoricalCrossEntropy
from aether.metrics import CategoricalAccuracy
from aether.model import Model
from aether.optimizers import Adam
from tests.base_case import BACKENDS_TO_TEST, register_test_suites
from tests.model.base import (
    ModelBaseTestCase,
    SpyLifecycleLayer,
    SpySyncAccuracy,
    SpySyncLoss,
    SpySyncOptimizer,
)


class TestModelConfig(ModelBaseTestCase):
    __test__ = False

    def test_configure_validation(self):
        """Test input validation across loss, optimizer, and metrics."""
        model = Model()

        with self.assertRaises(ValueError):
            model.configure()

        with self.assertRaises(TypeError):
            model.configure(loss=object())

        with self.assertRaises(TypeError):
            model.configure(optimizer=object())

        with self.assertRaises(TypeError):
            model.configure(accuracy=object())

    def test_sync_device_dispatches_on_finalize(self):
        """Verify finalize() calls _sync_device() to compile layers, loss, optimizer, and metrics."""
        model = Model()
        layer = SpyLifecycleLayer(self.NUM_FEATURES, self.NUM_CLASSES)
        loss = SpySyncLoss()
        opt = SpySyncOptimizer()
        acc = SpySyncAccuracy()

        model.add(layer)
        model.to(self.backend_name)
        model.configure(loss=loss, optimizer=opt, accuracy=acc)
        model.finalize((self.NUM_FEATURES,))

        self.assertEqual(layer.compiled_device, self.backend_name)
        self.assertEqual(loss.compiled_device, self.backend_name)
        self.assertEqual(opt.compiled_device, self.backend_name)
        self.assertEqual(acc.compiled_device, self.backend_name)

    def test_precision_policy_dispatch(self):
        """Verify set_precision() dispatches to non-exempt layers and preserves exempt layers."""
        model = Model()
        standard_layer = SpyLifecycleLayer(self.NUM_FEATURES, 8, precision_exempt=False)
        exempt_layer = SpyLifecycleLayer(8, self.NUM_CLASSES, precision_exempt=True)

        model.add(standard_layer)
        model.add(exempt_layer)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*NumPy float16 is emulated.*")
            model.set_precision("float16")

        self.assertIsNotNone(standard_layer.applied_policy)
        self.assertEqual(standard_layer.applied_policy.compute_dtype_name, "float16")
        self.assertIsNone(exempt_layer.applied_policy)

    def test_device_alignment_guard_raises_on_numpy_with_cupy_tensors(self):
        """Verify that a NumPy-configured model loudly rejects CuPy GPU inputs."""
        if "cupy" not in BACKENDS_TO_TEST:
            self.skipTest("CuPy backend not installed or configured in environment.")

        import cupy as cp

        model = Model()
        model.to("numpy")
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        model.configure(
            loss=SoftmaxCategoricalCrossEntropy(),
            optimizer=Adam(lr=0.01),
            accuracy=CategoricalAccuracy(),
        )
        model.finalize((self.NUM_FEATURES,))

        # Explicitly allocate GPU arrays via CuPy
        X_cupy = cp.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
        y_cupy = cp.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")

        with self.assertRaises(TypeError) as ctx:
            model.train(X_cupy, y_cupy, epochs=1, batch_size=8)
        self.assertIn("Device mismatch", str(ctx.exception))

        X_np = np.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
        y_np = np.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")
        with self.assertRaises(TypeError) as ctx:
            model.train(X_np, y_np, epochs=1, batch_size=8, validation_data=(X_cupy, y_cupy))
        self.assertIn("Device mismatch", str(ctx.exception))

    def test_device_alignment_guard_raises_on_cupy_with_numpy_tensors(self):
        """Verify that a CuPy-configured model loudly rejects host NumPy inputs."""
        if "cupy" not in BACKENDS_TO_TEST:
            self.skipTest("CuPy backend not installed or configured in environment.")

        import cupy as cp

        model = Model()
        model.to("cupy")
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        model.configure(
            loss=SoftmaxCategoricalCrossEntropy(),
            optimizer=Adam(lr=0.01),
            accuracy=CategoricalAccuracy(),
        )
        model.finalize((self.NUM_FEATURES,))

        # Explicitly allocate Host arrays via NumPy
        X_numpy = np.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
        y_numpy = np.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")

        with self.assertRaises(TypeError) as ctx:
            model.train(X_numpy, y_numpy, epochs=1, batch_size=8)
        self.assertIn("Device mismatch", str(ctx.exception))

        X_cp = cp.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
        y_cp = cp.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")
        with self.assertRaises(TypeError) as ctx:
            model.train(X_cp, y_cp, epochs=1, batch_size=8, validation_data=(X_numpy, y_numpy))
        self.assertIn("Device mismatch", str(ctx.exception))


register_test_suites(globals(), TestModelConfig)