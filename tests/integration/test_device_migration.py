import functools
import os
import tempfile
import warnings

import numpy as np

import aether as ae
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
    def _build_bn_model(self, *, fuse_kernels=True):
        """Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> LeakyReLU -> GAP -> Dense, in
        strict fp32: Conv2d's fp16 path would put its own rounding between the CPU and
        GPU results, and Adam's first step (about lr * sign(g)) would turn it into
        lr-sized parameter differences."""
        self.enterContext(warnings.catch_warnings())
        warnings.filterwarnings("ignore", message=r"\[aether\] Conv2d: precision policy requests 'float32'")
        model = ae.Model()
        model.add(ae.Conv2d(3, 8, (3, 3), (1, 1), padding="same"))
        model.add(ae.BatchNorm())
        model.add(ae.ReLU())
        model.add(ae.Conv2d(8, 8, (3, 3), (1, 1), padding="same"))
        model.add(ae.BatchNorm())
        model.add(ae.LeakyReLU(alpha=0.1))
        model.add(ae.GlobalAvgPool())
        model.add(ae.Dense(8, self.NUM_CLASSES))
        model.configure(
            loss=ae.SoftmaxCategoricalCrossEntropy(),
            optimizer=ae.Adam(lr=0.01),
        )
        model.to("cupy")
        model.manual_seed(seed=self.SEED)
        model.set_precision("float32")
        model.finalize(input_shape=(8, 8, 3), fuse_kernels=fuse_kernels)
        return model

    def _activations(self, layers):
        return [type(layer).__name__ for layer in layers if isinstance(layer, (ae.ReLU, ae.LeakyReLU))]

    def test_fused_batchnorm_activation_matches_unfused(self):
        """fuse_kernels=True skips the ReLU/LeakyReLU after each BatchNorm on CuPy, and
        gives the same outputs, gradients and optimizer step as fuse_kernels=False."""
        if not config.HAS_CUPY or self.backend_name != "cupy":
            self.skipTest("Kernel fusion only applies on the CuPy backend.")

        X, y = self.make_synthetic_image_data(n_samples=16, height=8, width=8)
        fused = self._build_bn_model(fuse_kernels=True)
        unfused = self._build_bn_model(fuse_kernels=False)
        self.assertEqual(self._activations(fused._forward_layers), [])
        self.assertEqual(self._activations(fused._backward_layers), [])
        self.assertEqual(self._activations(unfused._forward_layers), ["ReLU", "LeakyReLU"])

        outputs = [model.forward(X, training=True) for model in (fused, unfused)]
        self.assertTrue(self.xp.allclose(*outputs, atol=1e-5, rtol=1e-4))

        for model in (fused, unfused):
            model.train(X, y, epochs=1, verbose=0)
        for fused_layer, unfused_layer in zip(fused.trainable_layers, unfused.trainable_layers):
            # A bias in front of BatchNorm has a true gradient of 0 (BatchNorm removes the
            # mean), so its computed one is rounding noise that Adam scales up to ~lr.
            idx = fused.layers.index(fused_layer)
            before_norm = any(isinstance(layer, ae.BatchNorm) for layer in fused.layers[idx + 1:idx + 2])
            zero_grad = {"biases"} if before_norm else set()
            with self.subTest(layer=type(fused_layer).__name__):
                for name in ("dweights", "dbiases"):
                    self.assertTrue(self.xp.allclose(
                        getattr(fused_layer, name), getattr(unfused_layer, name), atol=1e-5, rtol=1e-4
                    ))
                for name, tensor in fused_layer.get_parameters().items():
                    if name in zero_grad:
                        continue
                    self.assertTrue(self.xp.allclose(
                        tensor, unfused_layer.get_parameters()[name], atol=1e-5, rtol=1e-4
                    ))

    def test_fused_model_runs_every_layer_after_to_numpy(self):
        """Regression guard: a model fused on CuPy must run its activations again after
        to('numpy'), whose BatchNorm doesn't apply them, and re-fuse on the way back."""
        if not config.HAS_CUPY or self.backend_name != "cupy":
            self.skipTest("Cross-backend migration requires CuPy to be active.")

        X, y = self.make_synthetic_image_data(n_samples=16, height=8, width=8)
        model = self._build_bn_model()
        model.train(X, y, epochs=1, verbose=0)
        gpu_logits = config.to_device(model.predict(X, return_logits=True), target="numpy")

        model.to("numpy")
        X_cpu = config.to_device(X, target="numpy")
        self.assertEqual(model._forward_layers, tuple(model.layers))
        cpu_logits = model.predict(X_cpu, return_logits=True)
        np.testing.assert_allclose(cpu_logits, gpu_logits, rtol=1e-4, atol=1e-4)
        self.assertGreaterEqual(model.layers[2].output.min(), 0.0)

        model.to("cupy")
        self.assertEqual(self._activations(model._forward_layers), [])
        regpu_logits = config.to_device(model.predict(X, return_logits=True), target="numpy")
        np.testing.assert_allclose(regpu_logits, gpu_logits, rtol=1e-4, atol=1e-4)

    def test_loaded_model_refuses_batchnorm_activation(self):
        """Fusion isn't saved: the manifest keeps every layer, and load() fuses again."""
        if not config.HAS_CUPY or self.backend_name != "cupy":
            self.skipTest("Kernel fusion only applies on the CuPy backend.")

        X, y = self.make_synthetic_image_data(n_samples=16, height=8, width=8)
        model = self._build_bn_model()
        model.train(X, y, epochs=1, verbose=0)
        expected = model.predict(X, return_logits=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "fused.aether")
            model.save(path)
            loaded = ae.Model.load(path, device="cupy")

        self.assertEqual(self._activations(loaded.layers), ["ReLU", "LeakyReLU"])
        self.assertEqual(self._activations(loaded._forward_layers), [])
        np.testing.assert_allclose(
            loaded.predict(X, return_logits=True), expected, rtol=1e-4, atol=1e-4
        )

register_test_suites(globals(), TestDeviceMigration)