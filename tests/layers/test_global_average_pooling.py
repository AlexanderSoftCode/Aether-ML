import numpy as np
import aether.config as config
import tests.base_case as base_case

from aether.layers.pooling import GlobalAvgPool

class TestGlobalAvgPool(base_case.AetherBaseLayerTestCase):
    S, H, W, C = 2, 28, 28, 3

    def setUp(self):
        super().setUp()
        self.layer = self.make_built_layer(
            GlobalAvgPool, 
            input_shape=(self.H, self.W, self.C),
        )
        self.test_images = self.xp.random.randn(self.S, self.H, self.W, self.C).astype(self.xp.float32)
        func_obj = getattr(self.layer.forward, 'func', self.layer.forward)
        self.uses_gpu_kernel = (getattr(func_obj, '__name__', '') == '_forward_gpu')

    def test_gap_forward_shape(self):
        """Verify output dimensions based on samples and channels"""

        output = self.layer.forward(self.test_images, training=True)
        expected_shape = (self.S, self.C)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_gpu_reduction_kernel_matches_cpu(self):
        
        if self.backend_name != 'cupy' or not config.HAS_CUPY:
            self.skipTest("Applies only to CuPy pass")

        layer_cpu = self.make_built_layer(
            GlobalAvgPool, 
            input_shape=(self.H, self.W, self.C),
        )
        layer_cpu._compile_for_device("numpy")
        cpu_output = layer_cpu.forward(self.test_images, training=True)

        gpu_inputs = self.xp.array(self.test_images)
        layer_gpu = self.make_built_layer(
            GlobalAvgPool, 
            input_shape=(self.H, self.W, self.C),
        )
        layer_gpu._compile_for_device("cupy")
        gpu_output = layer_gpu.forward(gpu_inputs, training=True)

        self.assertIsInstance(gpu_output, self.xp.ndarray)
        self.assertEqual(gpu_output.shape, (self.S, self.C))

        self.xp.testing.assert_allclose(
            self.xp.asnumpy(gpu_output),
            cpu_output,
            rtol=1e-5,
            atol=1e-5
        )

    def _gpu_layer(self, input_shape):
        if self.backend_name != 'cupy' or not config.HAS_CUPY:
            self.skipTest("Applies only to CuPy pass")
        layer = self.make_built_layer(GlobalAvgPool, input_shape=input_shape)
        layer._compile_for_device("cupy")
        return layer

    def test_forward_gpu_non_contiguous_and_non_float32_inputs(self):
        """Regression: the RawKernel read any buffer as dense NHWC float*."""
        base = self.xp.random.randn(4, 6, 5, 16).astype(self.xp.float32)
        cases = {
            "channel-strided view": base[..., ::2],
            "sample/height swapped view": base.transpose(1, 0, 2, 3),
            "float16": base.astype(self.xp.float16),
            "float64": base.astype(self.xp.float64),
        }
        for name, x in cases.items():
            with self.subTest(case=name):
                layer = self._gpu_layer(x.shape[1:])
                out = layer.forward(x, training=True)
                expected = x.astype(self.xp.float32).mean(axis=(1, 2))
                self.xp.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-3)

                dvalues = self.xp.ones_like(out)[:, ::-1]
                dinputs = layer.backward(dvalues)
                self.xp.testing.assert_allclose(
                    dinputs, self.xp.full(x.shape, 1.0 / (x.shape[1] * x.shape[2])), rtol=1e-5
                )

    def test_forward_gpu_after_conv_fallback(self):
        """Conv2d's fallback einsum output is not C-contiguous; feed it straight in."""
        from aether.layers.conv import Conv2d
        layer = self._gpu_layer((8, 8, 8))
        conv = self.make_built_layer(Conv2d, input_shape=(8, 8, 3), in_channels=3, out_channels=8)
        x = conv._forward_fallback(
            self.xp.random.randn(4, 8, 8, 3).astype(self.xp.float32), training=False
        )
        self.xp.testing.assert_allclose(
            layer.forward(x, training=False), x.mean(axis=(1, 2)), rtol=1e-5, atol=1e-5
        )

    def test_failed_kernel_compile_binds_fallback(self):
        """A kernel that failed to compile used to surface as 'NoneType' is not callable."""
        if self.backend_name != 'cupy' or not config.HAS_CUPY:
            self.skipTest("Applies only to CuPy pass")
        from unittest import mock
        from aether.custom_kernels import pooling_kernel
        layer = self.make_built_layer(GlobalAvgPool, input_shape=(self.H, self.W, self.C))
        with mock.patch.object(pooling_kernel, "get_gap_forward_kernel", return_value=None):
            layer._compile_for_device("cupy")
        self.assertIs(layer.forward.__func__, GlobalAvgPool._forward_fallback)
        self.assertIs(layer.backward.__func__, GlobalAvgPool._backward_fallback)

    def test_forward_known_values(self):
        """Verify mathematical output with uniform constant inputs."""
        constant_val = 5.5
        constant_input = self.xp.full((self.S, self.H, self.W, self.C), constant_val, dtype=self.xp.float32)
        
        out = self.layer.forward(constant_input, training=False)
        expected = self.xp.full((self.S, self.C), constant_val, dtype=self.xp.float32)
        
        self.xp.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_backward_gpu_kernel_matches_cpu_broadcast(self):
        """Cross-validate GPU backward kernel against CPU broadcasting fallback."""
        self.layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(self.S, self.C).astype(self.xp.float32)
        
        dinputs_actual = self.layer.backward(dvalues)
        
        inv_area = np.float32(1.0 / (self.H * self.W))
        dvalues4d = dvalues[:, self.xp.newaxis, self.xp.newaxis, :]
        dinputs_expected = self.xp.broadcast_to(dvalues4d, (self.S, self.H, self.W, self.C)) * inv_area
        
        self.xp.testing.assert_allclose(dinputs_actual, dinputs_expected, rtol=1e-5, atol=1e-6)

    def test_backward_gradient_sum_conservation(self):
        """Verify gradient conservation: sum(dinputs) per sample/channel equals dvalues."""
        self.layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(self.S, self.C).astype(self.xp.float32)
        
        dinputs = self.layer.backward(dvalues)
        
        # Summing dinputs across spatial axes (H, W) should yield back dvalues
        dinputs_sum = self.xp.sum(dinputs, axis=(1, 2))
        self.xp.testing.assert_allclose(dinputs_sum, dvalues, rtol=1e-5, atol=1e-6)

    def test_numerical_gradient_check(self):
        """Verify backward analytical gradient via centered finite differences."""
        S_small, H_small, W_small, C_small = 2, 4, 4, 2
        x_np = np.random.randn(S_small, H_small, W_small, C_small).astype(np.float32)
        x_xp = self.xp.array(x_np)
        
        small_layer = self.make_built_layer(
            GlobalAvgPool, 
            input_shape=(H_small, W_small, C_small),
        )
        small_layer.forward(x_xp, training=True)
        
        dvalues_np = np.random.randn(S_small, C_small).astype(np.float32)
        dvalues_xp = self.xp.array(dvalues_np)
        
        analytical_dinputs = small_layer.backward(dvalues_xp)
        if hasattr(analytical_dinputs, "get"):
            analytical_dinputs = analytical_dinputs.get()

        eps = 1e-3
        numerical_dinputs = np.zeros_like(x_np)

        # Compute numerical gradient for each input scalar element
        it = np.nditer(x_np, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            
            # x + eps
            x_pos = x_np.copy()
            x_pos[idx] += eps
            out_pos = small_layer.forward(self.xp.array(x_pos), training=False)
            if hasattr(out_pos, "get"): out_pos = out_pos.get()
            
            # x - eps
            x_neg = x_np.copy()
            x_neg[idx] -= eps
            out_neg = small_layer.forward(self.xp.array(x_neg), training=False)
            if hasattr(out_neg, "get"): out_neg = out_neg.get()
            
            # Scalar dot product with dvalues
            loss_pos = np.sum(out_pos * dvalues_np)
            loss_neg = np.sum(out_neg * dvalues_np)
            
            numerical_dinputs[idx] = (loss_pos - loss_neg) / (2.0 * eps)
            it.iternext()

        np.testing.assert_allclose(analytical_dinputs, numerical_dinputs, rtol=1e-3, atol=1e-4)

    def test_shape_meta_launch_cache(self):
        """Verify grid/block launch metadata caching for custom GPU kernels."""
        if not self.uses_gpu_kernel:
            self.skipTest("Skipping GPU launch metadata test for CPU backend.")
            
        shape = (4, 14, 14, 64)
        meta_1 = self.layer._get_shape_meta(shape)
        meta_2 = self.layer._get_shape_meta(shape)
        
        self.assertIs(meta_1, meta_2)
        self.assertEqual(meta_1["out_shape"], (4, 64))

    def test_thin_channels_launch_geometry(self):
        """Verify launch geometry calculation when channels C <= 32."""
        if not self.uses_gpu_kernel:
            self.skipTest("Skipping GPU launch metadata test for CPU backend.")

        thin_shape = (2, 28, 28, 8)
        meta = self.layer._get_shape_meta(thin_shape)
        
        self.assertEqual(meta["block_dim"][0], 8)
        self.assertEqual(meta["grid_dim"][0], 1)  # (8 + 8 - 1) // 8 == 1


base_case.register_test_suites(globals(), TestGlobalAvgPool)