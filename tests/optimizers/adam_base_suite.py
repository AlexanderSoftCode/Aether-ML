import aether.config as config
import tests.base_case as base_case
from aether.layers.conv import Conv2d
from aether.layers.linear import Dense
from aether.optimizers.adam import Adam


class BaseTestOptimizerAdam(base_case.AetherBaseLayerTestCase):
    __test__ = False

    OPTIMIZER_CLASS = Adam
    LR = 0.001
    DECAY = 0.0
    EPSILON = 1e-7
    BETA_1 = 0.9
    BETA_2 = 0.999
    N_INPUTS = 4
    N_NEURONS = 5

    def setUp(self):
        super().setUp()

        self.optimizer = self.make_component(
            self.OPTIMIZER_CLASS,
            lr=self.LR,
            decay=self.DECAY,
            epsilon=self.EPSILON,
            beta_1=self.BETA_1,
            beta_2=self.BETA_2,
        )

        self.layer = self.make_built_layer(
            Dense,
            input_shape=(self.N_INPUTS,),
            n_inputs=self.N_INPUTS,
            n_neurons=self.N_NEURONS,
        )
        self.layer.dweights = self.xp.random.randn(*self.layer.weights.shape).astype(
            self.xp.float32
        )
        self.layer.dbiases = self.xp.random.randn(*self.layer.biases.shape).astype(
            self.xp.float32
        )

        # Register layer buffers and bind device kernel
        self.optimizer.init_params([self.layer])
        if hasattr(self.optimizer, "_compile_for_device"):
            self.optimizer._compile_for_device(self.backend_name)

    def make_layer_and_optimizer(self, Layer_Class, Optimizer_Class, xp):
        layer = self.make_built_layer(
            Layer_Class,
            input_shape=(8, 8, 1),
            in_channels=1,
            out_channels=2,
            filter_size=(3, 3),
            stride=(1, 1),
            padding="same",
        )
        if hasattr(layer, "_compile_for_device"):
            layer._compile_for_device(self.backend_name)

        optimizer = self.make_component(
            Optimizer_Class,
            lr=0.01,
        )
        optimizer.init_params([layer])
        if hasattr(optimizer, "_compile_for_device"):
            optimizer._compile_for_device(self.backend_name)

        return layer, optimizer

    # ---- state initialization ---------------------------------------

    def test_init_params_allocates_momentum_and_cache_buffers(self):
        """init_params should pre-allocate weight/bias momentum and cache in float32."""
        fresh_optimizer = self.make_component(
            self.OPTIMIZER_CLASS,
            lr=self.LR,
        )
        fresh_layer = self.make_built_layer(
            Dense,
            input_shape=(self.N_INPUTS,),
            n_inputs=self.N_INPUTS,
            n_neurons=self.N_NEURONS,
        )

        self.assertFalse(hasattr(fresh_layer, "weight_momentums"))
        self.assertFalse(hasattr(fresh_layer, "weight_cache"))

        fresh_optimizer.init_params([fresh_layer])

        self.assertTrue(hasattr(fresh_layer, "weight_momentums"))
        self.assertTrue(hasattr(fresh_layer, "weight_cache"))
        self.assertTrue(hasattr(fresh_layer, "bias_momentums"))
        self.assertTrue(hasattr(fresh_layer, "bias_cache"))

        self.assertEqual(fresh_layer.weight_momentums.shape, fresh_layer.weights.shape)
        self.assertEqual(fresh_layer.weight_cache.shape, fresh_layer.weights.shape)
        self.assertEqual(fresh_layer.bias_momentums.shape, fresh_layer.biases.shape)
        self.assertEqual(fresh_layer.bias_cache.shape, fresh_layer.biases.shape)

        self.assertEqual(fresh_layer.weight_momentums.dtype, self.xp.float32)
        self.assertEqual(fresh_layer.weight_cache.dtype, self.xp.float32)
        self.assertEqual(fresh_layer.bias_momentums.dtype, self.xp.float32)
        self.assertEqual(fresh_layer.bias_cache.dtype, self.xp.float32)

    def test_step_accumulates_state_across_iterations(self):
        """Momentum/cache buffers should accumulate history across multiple step() calls."""
        self.optimizer.step()
        momentums_after_first = self.layer.weight_momentums.copy()

        self.layer.dweights = self.xp.random.randn(*self.layer.weights.shape).astype(
            self.xp.float32
        )
        self.layer.dbiases = self.xp.random.randn(*self.layer.biases.shape).astype(
            self.xp.float32
        )
        self.optimizer.step()

        reset_value = (1.0 - self.optimizer.beta_1) * self.layer.dweights
        self.assertFalse(
            self.xp.allclose(self.layer.weight_momentums, reset_value),
            msg="weight_momentums looks reinitialized rather than accumulated across steps.",
        )
        self.assertFalse(
            self.xp.allclose(momentums_after_first, self.layer.weight_momentums)
        )

    # ---- layer flexibility -------------------------------------------

    def test_biasless_layer_support(self):
        """Optimizer should update weights seamlessly without crashing when biases are absent."""
        biasless_layer = self.make_built_layer(
            Dense,
            input_shape=(self.N_INPUTS,),
            n_inputs=self.N_INPUTS,
            n_neurons=self.N_NEURONS,
        )
        biasless_layer.biases = None
        biasless_layer.dbiases = None
        biasless_layer.dweights = self.xp.ones_like(
            biasless_layer.weights, dtype=self.xp.float32
        )

        optimizer = self.make_component(
            self.OPTIMIZER_CLASS,
            lr=0.01,
        )
        optimizer.init_params([biasless_layer])
        if hasattr(optimizer, "_compile_for_device"):
            optimizer._compile_for_device(self.backend_name)

        self.assertFalse(hasattr(biasless_layer, "bias_momentums"))

        weights_before = biasless_layer.weights.copy()
        optimizer.step()
        self.assertFalse(self.xp.allclose(weights_before, biasless_layer.weights))

    # ---- learning rate decay & step progression ------------------------

    def test_step_applies_decay_and_increments_iterations(self):
        """step() should decay current_learning_rate and increment iterations atomically."""
        decayed_optimizer = self.make_component(
            self.OPTIMIZER_CLASS,
            lr=0.1,
            decay=0.01,
        )
        decayed_optimizer.init_params([self.layer])
        if hasattr(decayed_optimizer, "_compile_for_device"):
            decayed_optimizer._compile_for_device(self.backend_name)

        self.assertEqual(decayed_optimizer.iterations, 0)

        # Step 0: decay = 0.1 / (1 + 0.01 * 0) = 0.1, iterations incremented to 1
        decayed_optimizer.step()
        self.assertEqual(decayed_optimizer.iterations, 1)

        # Step 1: decay = 0.1 / (1 + 0.01 * 1) = 0.0990099
        decayed_optimizer.step()
        expected_lr = 0.1 * (1.0 / (1.0 + 0.01 * 1))
        self.assertAlmostEqual(
            decayed_optimizer.current_lr, expected_lr, places=6
        )
        self.assertEqual(decayed_optimizer.iterations, 2)

    # ---- update math correctness ---------------------------------------

    def test_update_matches_manual_adam_step(self):
        """Single-step check against textbook Adam update."""
        weights_before = self.layer.weights.copy()
        biases_before = self.layer.biases.copy()
        dweights = self.layer.dweights.copy()
        dbiases = self.layer.dbiases.copy()

        beta_1 = self.optimizer.beta_1
        beta_2 = self.optimizer.beta_2
        eps = self.optimizer.epsilon
        lr = self.optimizer.current_lr

        self.optimizer.step()

        m_w = (1 - beta_1) * dweights
        v_w = (1 - beta_2) * (dweights**2)
        m_w_hat = m_w / (1 - beta_1**1)
        v_w_hat = v_w / (1 - beta_2**1)
        expected_weights = weights_before - lr * m_w_hat / (
            self.xp.sqrt(v_w_hat) + eps
        )

        m_b = (1 - beta_1) * dbiases
        v_b = (1 - beta_2) * (dbiases**2)
        m_b_hat = m_b / (1 - beta_1**1)
        v_b_hat = v_b / (1 - beta_2**1)
        expected_biases = biases_before - lr * m_b_hat / (
            self.xp.sqrt(v_b_hat) + eps
        )

        self.xp.testing.assert_allclose(
            self.layer.weights, expected_weights, rtol=1e-5, atol=1e-6
        )
        self.xp.testing.assert_allclose(
            self.layer.biases, expected_biases, rtol=1e-5, atol=1e-6
        )

    def test_multi_step_descent_direction(self):
        start_weights = self.layer.weights.copy()

        for _ in range(10):
            self.layer.dweights = self.xp.ones(
                self.layer.weights.shape, dtype=self.xp.float32
            )
            self.layer.dbiases = self.xp.ones(
                self.layer.biases.shape, dtype=self.xp.float32
            )
            self.optimizer.step()

        self.assertTrue(self.xp.all(self.layer.weights < start_weights))

    def test_update_produces_finite_values(self):
        self.optimizer.step()
        self.assertTrue(self.xp.all(self.xp.isfinite(self.layer.weights)))
        self.assertTrue(self.xp.all(self.xp.isfinite(self.layer.biases)))

    # ---- in-place aliasing preservation --------------------------------

    def test_update_preserves_weight_buffer_identity(self):
        weights_id_before = id(self.layer.weights)
        biases_id_before = id(self.layer.biases)

        self.optimizer.step()

        self.assertEqual(weights_id_before, id(self.layer.weights))
        self.assertEqual(biases_id_before, id(self.layer.biases))

    # --- GPU & device cache regression tests ----------------------------

    def test_gpu_kernel_execution(self):
        if self.backend_name != "cupy":
            self.skipTest("GPU kernel test is only applicable to CuPy backend.")

        layer = self.make_built_layer(
            Dense,
            input_shape=(2,),
            n_inputs=2,
            n_neurons=2,
        )
        layer.dweights = self.xp.ones(layer.weights.shape, dtype=self.xp.float32)
        layer.dbiases = self.xp.ones(layer.biases.shape, dtype=self.xp.float32)

        optimizer = self.make_component(
            self.OPTIMIZER_CLASS,
            lr=0.01,
        )
        optimizer.init_params([layer])
        optimizer._compile_for_device("cupy")

        try:
            optimizer.step()
        except Exception as e:
            self.fail(f"CuPy kernel failed during optimizer.step(): {e}")

        self.assertTrue(self.xp.all(self.xp.isfinite(layer.weights)))
        self.assertTrue(self.xp.all(self.xp.isfinite(layer.biases)))

    def test_gpu_update_invalidates_fp16_weight_cache(self):
        if self.backend_name != "cupy":
            self.skipTest("GPU cache test is only applicable to CuPy backend.")

        layer, optimizer = self.make_layer_and_optimizer(
            Conv2d, self.OPTIMIZER_CLASS, self.xp
        )

        fixed_input = self.xp.random.randn(2, 8, 8, 1).astype(self.xp.float32)
        dvalues = self.xp.ones((2, 8, 8, 2), dtype=self.xp.float32)

        # Populate the layer's shadow cache on forward
        out_before = layer.forward(fixed_input, training=True).copy()
        layer.backward(dvalues)

        weights_before = layer.weights.copy()

        # Execute unified optimizer step
        optimizer.step()

        # 1. Shadow cache invalidation flag must be triggered
        self.assertFalse(
            layer._fp16_weight_valid,
            msg="invalidate_shadow_caches() was not called after the fused Adam kernel updated layer.weights",
        )

        # 2. In-place weights must have modified
        self.assertFalse(
            self.xp.allclose(weights_before, layer.weights),
            msg="weights did not change after optimizer.step()",
        )

        # 3. Output after step must reflect the newly updated weights
        out_after = layer.forward(fixed_input, training=False)
        self.assertFalse(
            self.xp.allclose(out_before, out_after),
            msg="forward() produced identical output after an optimizer step -- stale fp16 cache symptom",
        )