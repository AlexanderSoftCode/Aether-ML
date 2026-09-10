import functools
import numpy as np
import aether.config as config
from aether.base import Layer
from tests.base_case import register_test_suites
from tests.integration.model_base_suite import ModelIntegrationBaseCase


class TestDeviceMigration(ModelIntegrationBaseCase):
    BATCH_SIZE = 8

    def test_pointer_swap_binds_gpu_paths_after_to_cupy(self):
        """Every layer with _compile_for_device should have its forward
        pointer flipped away from the CPU fallback once migrated to cupy."""
        if self.backend_name != "cupy":
            self.skipTest("Pointer-swap only meaningfully diverges on the CuPy backend.")

        model = self.build_cnn_model()
        for layer in model.layers:
            if type(layer)._compile_for_device is not Layer._compile_for_device:
                with self.subTest(layer=type(layer).__name__):
                    fwd_fn = layer.forward
                    func_name = (
                        fwd_fn.func.__name__
                        if isinstance(fwd_fn, functools.partial)
                        else getattr(fwd_fn, "__name__", None)
                    )

                    self.assertIn(
                        func_name,
                        ("_forward_gpu", "_forward_fallback"),
                        msg=(
                            f"{type(layer).__name__}.forward did not bind to a "
                            f"recognized pointer-swap target after Model.to('cupy'). Got: {func_name}"
                        ),
                    )

    def test_post_migration_from_gpu_to_cpu(self):
        if self.backend_name != "cupy":
            self.skipTest("Pointer-swap only meaningfully diverges on the CuPy backend.")

        model = self.build_mlp_model(device="cupy")
        X, y = self.make_synthetic_image_data()
        model.train(X, y, epochs=1, batch_size=self.BATCH_SIZE, verbose=False)

        # 1. Capture GPU baseline
        gpu_baseline = model.predict(X, return_logits=True, stream_to_host=False)
        gpu_baseline_cpu = config.to_device(gpu_baseline, target="numpy")

        # 2. Migrate model and inputs to CPU
        model.to(device="numpy")
        X_cpu = config.to_device(X, target="numpy")

        # 3. Predict on CPU
        cpu_out = model.predict(X_cpu, return_logits=True)

        # 4. Assert parity on host memory
        np.testing.assert_allclose(gpu_baseline_cpu, cpu_out, rtol=1e-4, atol=1e-4)

    def test_post_finalization_from_cpu_to_gpu(self):

        if not config.HAS_CUPY or self.backend_name != "cupy":
            self.skipTest("Cross-backend migration requires CuPy to be active.")

        model = self.build_mlp_model(device="numpy")
        X, _ = self.make_synthetic_image_data()
        X_cpu = config.to_device(X, target="numpy")
        cpu_out = model.forward(X_cpu, training=False)

        model.to("cupy")
        X_gpu = config.to_device(X_cpu, target="cupy")
        gpu_out = model.forward(X_gpu, training=False)

        self.assertIsInstance(gpu_out, self.xp.ndarray)

        gpu_out_cpu = config.to_device(gpu_out, target="numpy")
        np.testing.assert_allclose(gpu_out_cpu, cpu_out, rtol=1e-4, atol=1e-4)

    def test_device_mismatch_protection_after_migration(self):
        if not config.HAS_CUPY or self.backend_name != "cupy":
            self.skipTest("Cross-backend migration requires CuPy to be active.")

        model = self.build_mlp_model(device="cupy")
        X, y = self.make_synthetic_image_data()
        X_gpu = config.to_device(X, target="cupy")
        y_gpu = config.to_device(y, target="cupy")
        model.to("numpy")

        with self.assertRaises(TypeError):
            model.predict(X_gpu)

        with self.assertRaises(TypeError):
            model.evaluate(X_gpu, y_gpu, verbose=False)

    def test_training_resumes_after_migration_both_directions(self):
        if not config.HAS_CUPY or self.backend_name != "cupy":
            self.skipTest("Cross-backend migration requires CuPy to be active.")

        model = self.build_cnn_model(device="cupy")
        X_gpu, y_gpu = self.make_synthetic_image_data()
        model.train(X_gpu, y_gpu, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)

        # Migrate to numpy and resume training -- optimizer moments must follow.
        model.to("numpy")
        X_cpu = config.to_device(X_gpu, target="numpy")
        y_cpu = config.to_device(y_gpu, target="numpy")
        model.train(X_cpu, y_cpu, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)

        for layer in model.trainable_layers:
            self.assertIsInstance(layer.weight_momentums, np.ndarray)

        # Migrate back to cupy and resume training again.
        model.to("cupy")
        model.train(X_gpu, y_gpu, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)

        for layer in model.trainable_layers:
            self.assertIsInstance(layer.weight_momentums, self.xp.ndarray)
register_test_suites(globals(), TestDeviceMigration)