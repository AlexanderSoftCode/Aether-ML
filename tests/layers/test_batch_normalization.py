import aether.config as config
import tests.base_case as base_case
from aether.layers.normalization import BatchNorm


class TestBatchNormLayer(base_case.AetherBaseLayerTestCase):
    NUM_FEATURES_DENSE = 16
    DENSE_SHAPE = (8, 16)               

    NUM_CHANNELS_CONV = 8
    CONV_SHAPE = (4, 14, 14, 8)

    # N = 4551 rows is not a multiple of the block's 8, C = 40 is not a multiple
    # of 32, and the reduction spans many blocks.
    RAGGED_SHAPE = (3, 37, 41, 40)

    EPSILON = 1e-5
    MOMENTUM = 0.9

    def setUp(self):
        super().setUp()

        # 2D Dense BatchNorm instance
        self.layer_dense = self.make_built_layer(
            BatchNorm, 
            input_shape=(self.DENSE_SHAPE[1],),
            epsilon=self.EPSILON, 
            momentum=self.MOMENTUM, 
        )

        # 4D Conv BatchNorm instance
        self.layer_conv = self.make_built_layer(
            BatchNorm, 
            input_shape=self.CONV_SHAPE[1:],
            epsilon=self.EPSILON, 
            momentum=self.MOMENTUM, 
        )

        # Seeded input batches
        self.xp.random.seed(42)
        self.dense_inputs = (
            self.xp.random.randn(*self.DENSE_SHAPE).astype(self.xp.float32) * 5.0 + 3.0
        )
        self.conv_inputs = (
            self.xp.random.randn(*self.CONV_SHAPE).astype(self.xp.float32) * 3.0 - 2.0
        )

    def test_build_parameter_initialization(self):
        """Verify gamma, beta, running mean, and running variance allocation."""
        # Trainable parameters
        self.assertEqual(self.layer_dense.gamma.shape, (self.NUM_FEATURES_DENSE,))
        self.assertEqual(self.layer_dense.beta.shape, (self.NUM_FEATURES_DENSE,))
        self.assertTrue(self.xp.allclose(self.layer_dense.gamma, 1.0))
        self.assertTrue(self.xp.allclose(self.layer_dense.beta, 0.0))

        # Optimizer aliases
        self.assertIs(self.layer_dense.weights, self.layer_dense.gamma)
        self.assertIs(self.layer_dense.biases, self.layer_dense.beta)

        # Non-trainable running statistics
        self.assertEqual(self.layer_dense.running_mean.shape, (self.NUM_FEATURES_DENSE,))
        self.assertEqual(self.layer_dense.running_var.shape, (self.NUM_FEATURES_DENSE,))
        self.assertTrue(self.xp.allclose(self.layer_dense.running_mean, 0.0))
        self.assertTrue(self.xp.allclose(self.layer_dense.running_var, 1.0))

    def test_forward_training_normalization_2d(self):
        """Verify that 2D batch outputs are normalized to zero mean and unit variance."""
        out = self.layer_dense.forward(self.dense_inputs, training=True)

        self.assertEqual(out.shape, self.DENSE_SHAPE)
        
        # Batch mean across batch dimension should be close to 0 (beta=0)
        batch_mean = self.xp.mean(out, axis=0)
        self.assertTrue(self.xp.allclose(batch_mean, 0.0, atol=1e-5))

        # Batch variance across batch dimension should be close to 1 (gamma=1)
        batch_var = self.xp.var(out, axis=0)
        self.assertTrue(self.xp.allclose(batch_var, 1.0, atol=1e-3))

    def test_forward_training_normalization_4d(self):
        """Verify that 4D CNN outputs are normalized across spatial and batch axes."""
        out = self.layer_conv.forward(self.conv_inputs, training=True)

        self.assertEqual(out.shape, self.CONV_SHAPE)
        
        batch_mean = self.xp.mean(out, axis=(0, 1, 2))
        batch_var = self.xp.var(out, axis=(0, 1, 2))

        self.assertTrue(self.xp.allclose(batch_mean, 0.0, atol=1e-4))
        self.assertTrue(self.xp.allclose(batch_var, 1.0, atol=1e-3))

    def test_running_statistics_ema_update(self):
        """Verify the exponential moving average update for running statistics."""
        initial_mean = self.layer_dense.running_mean.copy()
        initial_var = self.layer_dense.running_var.copy()

        batch_mean = self.xp.mean(self.dense_inputs, axis=0)
        batch_var = self.xp.var(self.dense_inputs, axis=0)

        self.layer_dense.forward(self.dense_inputs, training=True)

        expected_running_mean = (
            self.MOMENTUM * initial_mean + (1.0 - self.MOMENTUM) * batch_mean
        )
        expected_running_var = (
            self.MOMENTUM * initial_var + (1.0 - self.MOMENTUM) * batch_var
        )

        self.assertTrue(self.xp.allclose(self.layer_dense.running_mean, expected_running_mean))
        self.assertTrue(self.xp.allclose(self.layer_dense.running_var, expected_running_var))

    def test_forward_inference_mode(self):
        """Verify inference mode uses running stats and does not mutate them or store caches."""
        # Populate running statistics with custom values
        test_mean = self.xp.full((self.NUM_FEATURES_DENSE,), 3.0, dtype=self.xp.float32)
        test_var = self.xp.full((self.NUM_FEATURES_DENSE,), 4.0, dtype=self.xp.float32)
        self.layer_dense.running_mean = test_mean.copy()
        self.layer_dense.running_var = test_var.copy()

        out = self.layer_dense.forward(self.dense_inputs, training=False)

        # Check mathematical result: (x - test_mean) / sqrt(test_var + eps)
        expected = (self.dense_inputs - 3.0) / self.xp.sqrt(4.0 + self.EPSILON)
        self.assertTrue(self.xp.allclose(out, expected, atol=1e-5))

        # Running statistics must remain untouched
        self.assertTrue(self.xp.allclose(self.layer_dense.running_mean, test_mean))
        self.assertTrue(self.xp.allclose(self.layer_dense.running_var, test_var))

    def test_backward_gradients_shapes_2d(self):
        """Verify gradient shapes for 2D Dense input backpropagation."""
        self.layer_dense.forward(self.dense_inputs, training=True)

        dvalues = self.xp.ones_like(self.dense_inputs)
        dinputs = self.layer_dense.backward(dvalues)

        self.assertEqual(dinputs.shape, self.DENSE_SHAPE)
        self.assertEqual(self.layer_dense.dweights.shape, (self.NUM_FEATURES_DENSE,))
        self.assertEqual(self.layer_dense.dbiases.shape, (self.NUM_FEATURES_DENSE,))

    def test_backward_gradients_shapes_4d(self):
        """Verify gradient shapes for 4D CNN input backpropagation."""
        self.layer_conv.forward(self.conv_inputs, training=True)

        dvalues = self.xp.ones_like(self.conv_inputs)
        dinputs = self.layer_conv.backward(dvalues)

        self.assertEqual(dinputs.shape, self.CONV_SHAPE)
        self.assertEqual(self.layer_conv.dweights.shape, (self.NUM_CHANNELS_CONV,))
        self.assertEqual(self.layer_conv.dbiases.shape, (self.NUM_CHANNELS_CONV,))

    def test_precision_retention_float32(self):
        """Verify calculations do not promote float32 arrays to float64."""
        out = self.layer_conv.forward(self.conv_inputs, training=True)
        dinputs = self.layer_conv.backward(self.xp.ones_like(out))

        self.assertEqual(out.dtype, self.xp.float32)
        self.assertEqual(dinputs.dtype, self.xp.float32)
        self.assertEqual(self.layer_conv.running_mean.dtype, self.xp.float32)
        self.assertEqual(self.layer_conv.running_var.dtype, self.xp.float32)
        self.assertEqual(self.layer_conv.dweights.dtype, self.xp.float32)
        self.assertEqual(self.layer_conv.dbiases.dtype, self.xp.float32)

    # ---- GPU tests --------------------------

    def test_gpu_launch_cache_population(self):
        """Verify that _launch_cache correctly populates and memoizes metadata."""
        if self.backend_name != "cupy":
            self.skipTest("GPU launch cache test is only applicable on CuPy backend.")

        # Ensure cache is initially empty before forward passes
        self.assertEqual(len(self.layer_dense._launch_cache), 0)

        # Forward pass should populate cache for DENSE_SHAPE
        self.layer_dense.forward(self.dense_inputs, training=True)
        self.assertIn(self.DENSE_SHAPE, self.layer_dense._launch_cache)

        meta = self.layer_dense._launch_cache[self.DENSE_SHAPE]
        self.assertEqual(meta["N"], self.DENSE_SHAPE[0])
        self.assertEqual(meta["C"], self.DENSE_SHAPE[1])
        self.assertEqual(meta["block_dim"], (32, 8, 1))
        self.assertEqual(meta["grid_dim"], ((self.DENSE_SHAPE[1] + 31) // 32, 1, 1))

        # Repeated execution preserves cache size without re-allocation
        self.layer_dense.forward(self.dense_inputs, training=True)
        self.assertEqual(len(self.layer_dense._launch_cache), 1)

    def _make_ragged(self):
        """A RAGGED_SHAPE layer with non-trivial gamma/beta, so a scale/inv_std mix-up
        can't hide behind gamma = 1, plus a matching input batch."""
        C = self.RAGGED_SHAPE[-1]
        layer = self.make_built_layer(
            BatchNorm, input_shape=self.RAGGED_SHAPE[1:], epsilon=self.EPSILON, momentum=self.MOMENTUM
        )
        layer.gamma = self.xp.random.uniform(0.5, 2.0, C).astype(self.xp.float32)
        layer.beta = self.xp.random.randn(C).astype(self.xp.float32)
        inputs = self.xp.random.randn(*self.RAGGED_SHAPE).astype(self.xp.float32) * 2.0 + 5.0
        return layer, inputs

    def test_gpu_forward_training_parity(self):
        """Verify parity between fallback and GPU forward training implementations."""
        if self.backend_name != "cupy":
            self.skipTest("GPU parity test is only applicable on CuPy backend.")

        for layer, inputs in [(self.layer_conv, self.conv_inputs), self._make_ragged()]:
            C = inputs.shape[-1]
            with self.subTest(shape=inputs.shape):
                # Execute fallback path
                out_fallback = layer._forward_fallback(inputs, training=True)
                mean_fallback = layer.batch_mean.copy()
                var_fallback = layer.batch_var.copy()
                running_mean_fallback = layer.running_mean.copy()
                running_var_fallback = layer.running_var.copy()

                # Reset parameters and running statistics to identical initial state
                layer.running_mean = self.xp.zeros(C, dtype=self.xp.float32)
                layer.running_var = self.xp.ones(C, dtype=self.xp.float32)

                # Execute GPU path
                out_gpu = layer._forward_gpu(inputs, training=True)

                self.assertTrue(self.xp.allclose(out_gpu, out_fallback, atol=1e-4, rtol=1e-4))
                self.assertTrue(self.xp.allclose(layer.batch_mean, mean_fallback.squeeze(), atol=1e-4, rtol=1e-4))
                self.assertTrue(self.xp.allclose(layer.batch_var, var_fallback.squeeze(), atol=1e-4, rtol=1e-4))
                self.assertTrue(self.xp.allclose(layer.running_mean, running_mean_fallback, atol=1e-4, rtol=1e-4))
                self.assertTrue(self.xp.allclose(layer.running_var, running_var_fallback, atol=1e-4, rtol=1e-4))

    def test_gpu_forward_inference_parity(self):
        """Verify parity between fallback and GPU forward inference paths."""
        if self.backend_name != "cupy":
            self.skipTest("GPU parity test is only applicable on CuPy backend.")

        custom_mean = self.xp.full((self.NUM_CHANNELS_CONV,), 1.5, dtype=self.xp.float32)
        custom_var = self.xp.full((self.NUM_CHANNELS_CONV,), 2.5, dtype=self.xp.float32)

        self.layer_conv.running_mean = custom_mean.copy()
        self.layer_conv.running_var = custom_var.copy()
        out_fallback = self.layer_conv._forward_fallback(self.conv_inputs, training=False)

        out_gpu = self.layer_conv._forward_gpu(self.conv_inputs, training=False)
        self.assertTrue(self.xp.allclose(out_gpu, out_fallback, atol=1e-4, rtol=1e-4))

    def test_gpu_backward_parity_2d_and_4d(self):
        """Verify gradient parity across dinputs, dweights, and dbiases for 2D and 4D."""
        if self.backend_name != "cupy":
            self.skipTest("GPU parity test is only applicable on CuPy backend.")

        for layer, inputs in [
            (self.layer_dense, self.dense_inputs),
            (self.layer_conv, self.conv_inputs),
            self._make_ragged(),
        ]:
            dvalues = self.xp.random.randn(*inputs.shape).astype(self.xp.float32)

            # Fallback path run
            layer._forward_fallback(inputs, training=True)
            dinputs_fallback = layer._backward_fallback(dvalues)
            dweights_fallback = layer.dweights.copy()
            dbiases_fallback = layer.dbiases.copy()

            # GPU path run
            layer._forward_gpu(inputs, training=True)
            dinputs_gpu = layer._backward_gpu(dvalues)

            with self.subTest(shape=inputs.shape):
                self.assertTrue(self.xp.allclose(dinputs_gpu, dinputs_fallback, atol=1e-4, rtol=1e-4))
                self.assertTrue(self.xp.allclose(layer.dweights, dweights_fallback, atol=1e-4, rtol=1e-4))
                self.assertTrue(self.xp.allclose(layer.dbiases, dbiases_fallback, atol=1e-4, rtol=1e-4))

    def test_gpu_training_is_deterministic(self):
        """The reductions write per-block partials instead of float atomics, so two
        runs over many blocks give bit-identical statistics, outputs and gradients."""
        if self.backend_name != "cupy":
            self.skipTest("GPU determinism test is only applicable on CuPy backend.")

        layer, inputs = self._make_ragged()
        dvalues = self.xp.random.randn(*inputs.shape).astype(self.xp.float32)

        runs = []
        for _ in range(2):
            out = layer._forward_gpu(inputs, training=True)
            stats = [layer.batch_mean.copy(), layer.batch_var.copy()]
            dinputs = layer._backward_gpu(dvalues)
            runs.append([out.copy(), *stats, dinputs.copy(), layer.dweights.copy(), layer.dbiases.copy()])

        for name, first, second in zip(
            ("output", "batch_mean", "batch_var", "dinputs", "dweights", "dbiases"), *runs
        ):
            with self.subTest(name=name):
                self.assertTrue(self.xp.array_equal(first, second))

    def test_gpu_float64_running_stats(self):
        """Regression: training on NumPy under a float64 policy leaves the running
        statistics float64, and Model.to('cupy') keeps that dtype. GPU inference then
        raised TypeError, and the in-place running-stat update must not misread them."""
        if self.backend_name != "cupy":
            self.skipTest("GPU dtype test is only applicable on CuPy backend.")

        layer = self.layer_dense
        mean64 = self.xp.linspace(-1.0, 1.0, self.NUM_FEATURES_DENSE)
        var64 = self.xp.linspace(0.5, 2.0, self.NUM_FEATURES_DENSE)

        layer.running_mean, layer.running_var = mean64.copy(), var64.copy()
        layer._forward_gpu(self.dense_inputs, training=True)
        expected_mean = 0.9 * mean64 + 0.1 * self.xp.mean(self.dense_inputs, axis=0)
        expected_var = 0.9 * var64 + 0.1 * self.xp.var(self.dense_inputs, axis=0)
        self.assertTrue(self.xp.allclose(layer.running_mean, expected_mean, atol=1e-5, rtol=1e-5))
        self.assertTrue(self.xp.allclose(layer.running_var, expected_var, atol=1e-5, rtol=1e-5))

        layer.running_mean, layer.running_var = mean64.copy(), var64.copy()
        out = layer._forward_gpu(self.dense_inputs, training=False)
        expected = (self.dense_inputs - mean64) / self.xp.sqrt(var64 + self.EPSILON)
        self.assertTrue(self.xp.allclose(out, expected, atol=1e-5, rtol=1e-5))

    def test_gpu_non_float32_inputs_match_float32(self):
        """Regression: under set_precision() a Dense before BatchNorm emits fp16/fp64,
        which the float* reduction kernels misread and the fp32-only elementwise
        kernel rejected (TypeError) after running_mean was already corrupted."""
        if self.backend_name != "cupy":
            self.skipTest("GPU dtype test is only applicable on CuPy backend.")

        dvalues = self.xp.random.randn(*self.DENSE_SHAPE).astype(self.xp.float32)
        for dtype in (self.xp.float16, self.xp.float64):
            with self.subTest(dtype=dtype.__name__):
                inputs = self.dense_inputs.astype(dtype)
                reference = inputs.astype(self.xp.float32)

                layer = self.layer_dense
                expected_out = layer._forward_fallback(reference, training=True)
                expected_dinputs = layer._backward_fallback(dvalues)
                layer.running_mean = self.xp.zeros(self.NUM_FEATURES_DENSE, dtype=self.xp.float32)
                layer.running_var = self.xp.ones(self.NUM_FEATURES_DENSE, dtype=self.xp.float32)
                expected_running_mean = 0.1 * self.xp.mean(reference, axis=0)

                out = layer._forward_gpu(inputs, training=True)
                dinputs = layer._backward_gpu(dvalues.astype(dtype))

                self.assertTrue(self.xp.allclose(out, expected_out, atol=1e-4, rtol=1e-4))
                self.assertTrue(self.xp.allclose(dinputs, expected_dinputs, atol=1e-3, rtol=1e-3))
                self.assertTrue(self.xp.allclose(
                    layer.running_mean, expected_running_mean, atol=1e-4, rtol=1e-4
                ))
                layer._forward_gpu(inputs, training=False)

    def test_gpu_numerical_stability_large_offset(self):
        """Verify variance stability under large baseline offsets and low variance."""
        if self.backend_name != "cupy":
            self.skipTest("GPU numerical stability test is only applicable on CuPy backend.")

        # Inputs with large mean offset and tiny variance
        offset_inputs = (
            self.xp.random.randn(*self.CONV_SHAPE).astype(self.xp.float32) * 0.01 + 10000.0
        )

        out = self.layer_conv._forward_gpu(offset_inputs, training=True)

        # E[x^2] - E[x]^2 in fp32 cancels to noise here; the reference is the float64
        # variance of the same fp32 data.
        expected_var = self.xp.var(offset_inputs.astype(self.xp.float64), axis=(0, 1, 2))
        self.assertTrue(self.xp.allclose(self.layer_conv.batch_var, expected_var, atol=0, rtol=1e-4))
        self.assertTrue(self.xp.all(self.layer_conv.batch_var >= 0.0))
        self.assertFalse(self.xp.any(self.xp.isnan(out)))
        self.assertFalse(self.xp.any(self.xp.isinf(out)))

    # ---- inference tests ----------------------------
    
    def test_forward_inference_affine_folding_math(self):
        """Verify the fused affine (scale * x + bias) formulation matches standard normalization."""
        # Custom non-trivial parameters and running statistics
        self.layer_conv.gamma = self.xp.array([0.5, 1.2, -0.8, 2.0, 0.1, 1.5, -1.0, 0.7], dtype=self.xp.float32)
        self.layer_conv.beta = self.xp.array([0.1, -0.2, 0.5, 1.0, -1.0, 0.3, 0.0, -0.5], dtype=self.xp.float32)
        self.layer_conv.running_mean = self.xp.array([1.0, -2.0, 0.5, 3.0, -1.5, 0.0, 2.5, -0.5], dtype=self.xp.float32)
        self.layer_conv.running_var = self.xp.array([2.0, 1.5, 0.8, 4.0, 0.2, 1.0, 3.0, 0.5], dtype=self.xp.float32)

        out = self.layer_conv.forward(self.conv_inputs, training=False)

        # Expected manual calculation via fused affine parameters
        inv_std = 1.0 / self.xp.sqrt(self.layer_conv.running_var + self.EPSILON)
        expected_scale = self.layer_conv.gamma * inv_std
        expected_bias = self.layer_conv.beta - self.layer_conv.running_mean * expected_scale
        expected_out = self.conv_inputs * expected_scale + expected_bias

        self.assertTrue(self.xp.allclose(out, expected_out, atol=1e-5, rtol=1e-5))

    def test_gpu_forward_inference_path_active(self):
        """Verify that training=False on GPU uses the fused affine kernel without triggering reductions."""
        if self.backend_name != "cupy":
            self.skipTest("GPU inference kernel test is only applicable on CuPy backend.")

        # Ensure launch cache starts clean
        self.layer_conv._launch_cache.clear()

        # Set custom test stats
        self.layer_conv.running_mean = self.xp.full((self.NUM_CHANNELS_CONV,), 2.0, dtype=self.xp.float32)
        self.layer_conv.running_var = self.xp.full((self.NUM_CHANNELS_CONV,), 3.0, dtype=self.xp.float32)
        initial_running_mean = self.layer_conv.running_mean.copy()
        initial_running_var = self.layer_conv.running_var.copy()

        # Execute GPU inference forward pass
        out_gpu = self.layer_conv._forward_gpu(self.conv_inputs, training=False)

        # 1. Reduction metadata should NOT be cached during inference
        self.assertEqual(len(self.layer_conv._launch_cache), 0)

        # 2. Running statistics must remain completely un-mutated
        self.assertTrue(self.xp.allclose(self.layer_conv.running_mean, initial_running_mean))
        self.assertTrue(self.xp.allclose(self.layer_conv.running_var, initial_running_var))

        # 3. Intermediate backward training caches must NOT be stored
        self.assertIsNone(self.layer_conv.batch_mean)
        self.assertIsNone(self.layer_conv.batch_var)
        self.assertIsNone(self.layer_conv.normalized)
        self.assertIsNone(self.layer_conv.inv_std)

        # 4. Numerical output must match the explicit vectorized broadcast formula
        inv_std = 1.0 / self.xp.sqrt(initial_running_var + self.EPSILON)
        scale = self.layer_conv.gamma * inv_std
        bias = self.layer_conv.beta - initial_running_mean * scale
        expected = self.conv_inputs * scale + bias
        self.assertTrue(self.xp.allclose(out_gpu, expected, atol=1e-5, rtol=1e-5))

    def test_gpu_vs_fallback_inference_parity(self):
        """Verify exact output parity between fallback and GPU inference across 2D and 4D."""
        if self.backend_name != "cupy":
            self.skipTest("GPU parity test is only applicable on CuPy backend.")

        for layer, inputs, num_features in [
            (self.layer_dense, self.dense_inputs, self.NUM_FEATURES_DENSE),
            (self.layer_conv, self.conv_inputs, self.NUM_CHANNELS_CONV),
        ]:
            # Seed running stats with non-trivial values
            layer.running_mean = self.xp.random.randn(num_features).astype(self.xp.float32)
            layer.running_var = self.xp.abs(self.xp.random.randn(num_features)).astype(self.xp.float32) + 0.5
            layer.gamma = self.xp.random.randn(num_features).astype(self.xp.float32)
            layer.beta = self.xp.random.randn(num_features).astype(self.xp.float32)

            out_fallback = layer._forward_fallback(inputs, training=False)
            out_gpu = layer._forward_gpu(inputs, training=False)

            self.assertTrue(self.xp.allclose(out_gpu, out_fallback, atol=1e-5, rtol=1e-5))

base_case.register_test_suites(globals(), TestBatchNormLayer)