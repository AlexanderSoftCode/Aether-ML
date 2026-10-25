import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.layers.linear import Dense


class TestDenseLayer(base_case.AetherBaseLayerTestCase): 
    SEED = 42
    N_INPUTS = 8
    N_NEURONS = 4
    BATCH_SIZE = 3

    def setUp(self):
        super().setUp()
        self.xp.random.seed(self.SEED)

        self.layer = self.make_built_layer(
            Dense,
            input_shape=(self.N_INPUTS,),
            seed=self.SEED, 
            n_inputs=self.N_INPUTS, 
            n_neurons=self.N_NEURONS)
        self.test_inputs = self.xp.random.randn(self.BATCH_SIZE, self.N_INPUTS).astype(self.xp.float32) 

    def test_forward_shape(self):
        "Verify output shape is 10, 3"
        output = self.layer.forward(self.test_inputs, training=True)
        self.assertEqual(output.shape, (self.BATCH_SIZE, self.N_NEURONS))

    def test_parameters_initialization(self):
        """Verify parameter buffer shapes, float32 defaults, and He Initialization"""
        self.assertEqual(self.layer.weights.shape, (self.N_INPUTS, self.N_NEURONS))
        self.assertEqual(self.layer.biases.shape, (1, self.N_NEURONS))
        self.assertEqual(self.layer.weights.dtype, self.xp.float32)
        self.assertEqual(self.layer.biases.dtype, self.xp.float32)

        self.assertTrue(self.xp.all(self.layer.biases == 0))

        # He initialization scale check
        large_layer = self.make_built_layer(Dense, input_shape=(256,), n_inputs=256, n_neurons=256)
        expected_std = float(self.xp.sqrt(2.0 / 256))
        actual_std = float(self.xp.std(large_layer.weights))
        self.assertAlmostEqual(actual_std, expected_std, delta=0.03)

    def test_dense_zero_input(self):
        "Given an input of all zeros, show an output of all zeros (plus biases)"

        layer = self.make_built_layer(Dense, input_shape=(728,), n_inputs=728, n_neurons=20)
        
        zero_input = self.xp.zeros((1, 728))

        # Every pixel in the output should equal the bias for that layer
        # We reshape the biases to (1, biases) to match output shape

        output = layer.forward(zero_input, training=False)
        expected_output = self.xp.broadcast_to(layer.biases, output.shape)

        # compare the values
        self.xp.testing.assert_array_almost_equal(output, expected_output)

    def test_forward_training_branch_caching(self):
        """Verify forward(training=True) populates ephemeral compute caches."""
        self.layer.forward(self.test_inputs, training=True)
        self.assertIsNotNone(self.layer.inputs)
        self.assertIsNotNone(self.layer._inputs_compute)
        self.assertIsNotNone(self.layer._weights_compute)

    def test_forward_inference_branch_clears_cache(self):
        """Verify forward(training=False) clears compute caches."""
        self.layer.forward(self.test_inputs, training=True)
        self.layer.forward(self.test_inputs, training=False)

        self.assertIsNone(self.layer.inputs)
        self.assertIsNone(self.layer._inputs_compute)
        self.assertIsNone(self.layer._weights_compute)

    def test_backward_without_training_forward_raises_runtime_error(self):

        dvalues = self.xp.random.randn(self.BATCH_SIZE, self.N_NEURONS).astype(self.xp.float32)

        self.layer.forward(self.test_inputs, training=False)
        with self.assertRaises(RuntimeError):
            self.layer.backward(dvalues)

    def test_backward_gradient_shapes_and_cache_cleanup(self):
        """Ensure backprop computes valid gradient shapes and invalidates cache."""

        output = self.layer.forward(self.test_inputs, training=True)
        dvalues = self.xp.random.randn(*output.shape).astype(self.xp.float32)
        dinputs = self.layer.backward(dvalues)

        self.assertEqual(dinputs.shape, self.test_inputs.shape)
        self.assertEqual(self.layer.dweights.shape, self.layer.weights.shape)
        self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)

        # Caches cleared post-backward
        self.assertIsNone(self.layer._inputs_compute)
        self.assertIsNone(self.layer._weights_compute)

    def test_mixed_precision_flow_float16(self):
        """Verify forward compute casting and gradient recovery to float32."""
        self.set_precision(self.layer, compute_dtype='float16')

        output = self.layer.forward(self.test_inputs, training=True)
        self.assertEqual(output.dtype, self.xp.float16)

        dvalues = self.xp.random.randn(*output.shape).astype(self.xp.float32)
        dinputs = self.layer.backward(dvalues)

        # Gradients to previous layer match input precision (float32)
        self.assertEqual(dinputs.dtype, self.test_inputs.dtype)
        # Stored master parameter gradients cast back to param precision (float32)
        self.assertEqual(self.layer.dweights.dtype, self.xp.float32)
        self.assertEqual(self.layer.dbiases.dtype, self.xp.float32)

    def test_numerical_gradient_check(self):
        """Finite difference verification of analytical dweights and dinputs."""
        layer = self.make_built_layer(Dense, input_shape=(4,), n_inputs=4, n_neurons=3)
        layer.weights = layer.weights.astype(self.xp.float64)
        layer.biases = layer.biases.astype(self.xp.float64)

        x = self.xp.random.randn(2, 4).astype(self.xp.float64)
        dvalues = self.xp.random.randn(2, 3).astype(self.xp.float64)
        eps = 1e-6

        # Analytical gradients
        layer.forward(x, training=True)
        layer.backward(dvalues)
        analytical_dweights = layer.dweights.copy()

        # Numerical gradients for weights
        numerical_dweights = self.xp.zeros_like(layer.weights)
        it = np.nditer(config.to_device(layer.weights, 'numpy'), flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index

            layer.weights[idx] += eps
            out_pos = layer.forward(x, training=True)
            loss_pos = self.xp.sum(out_pos * dvalues)

            layer.weights[idx] -= 2 * eps
            out_neg = layer.forward(x, training=True)
            loss_neg = self.xp.sum(out_neg * dvalues)

            layer.weights[idx] += eps  # restore
            numerical_dweights[idx] = (loss_pos - loss_neg) / (2 * eps)
            it.iternext()

        rel_error = self.xp.max(
            self.xp.abs(analytical_dweights - numerical_dweights)
            / (self.xp.abs(analytical_dweights) + self.xp.abs(numerical_dweights) + 1e-8)
        )
        self.assertLess(float(rel_error), 1e-4)

    # ----Dense.build() Seed Tests ----------

    def test_same_seed_produces_identical_weights(self):
        """Two independently instantiated layers built with the exact same seed
        must initialize with identical weight and bias values."""
        layer_1 = Dense(n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS)
        layer_2 = Dense(n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS)

        seed_val = 123
        layer_1.build(input_shape=(self.N_INPUTS,), seed=seed_val)
        layer_2.build(input_shape=(self.N_INPUTS,), seed=seed_val)

        # Weights and biases must match exactly bit-for-bit on the active backend
        self.xp.testing.assert_array_equal(
            layer_1.weights, layer_2.weights,
            err_msg="Layers built with identical seeds produced diverging weights."
        )
        self.xp.testing.assert_array_equal(
            layer_1.biases, layer_2.biases,
            err_msg="Layers built with identical seeds produced diverging biases."
        )

    def test_different_seeds_produce_divergent_weights(self):
        """Two layers built with different seeds must produce different parameter initializations."""
        layer_1 = Dense(n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS)
        layer_2 = Dense(n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS)

        layer_1.build(input_shape=(self.N_INPUTS,), seed=123)
        layer_2.build(input_shape=(self.N_INPUTS,), seed=456)

        # Weights should not be identical
        self.assertFalse(
            self.xp.array_equal(layer_1.weights, layer_2.weights),
            msg="Layers built with different seeds produced identical weights."
        )

    def test_build_seed_does_not_corrupt_global_rng_stream(self):
        """Calling build(seed=...) must draw from a local PRNG and avoid resetting
        or corrupting the backend's ambient global random state."""
        xp = self.xp

        # Seed global backend state and sample a reference number
        xp.random.seed(999)
        val_before = float(xp.random.randn(1)[0])

        # Reset global state to 999, build an intermediate layer with an isolated seed,
        # and verify the next global draw is unaffected
        xp.random.seed(999)
        throwaway_layer = Dense(n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS)
        throwaway_layer.build(input_shape=(self.N_INPUTS,), seed=123)

        val_after = float(xp.random.randn(1)[0])

        self.assertAlmostEqual(
            val_before, val_after, places=6,
            msg="layer.build(seed=...) modified or reset the global backend random state."
        )

    def test_build_with_none_seed_uses_global_stream(self):
        """Calling build(seed=None) should sequentially consume random numbers
        from the global stream rather than throwing an error or using a static seed."""
        xp = self.xp

        xp.random.seed(777)
        layer_1 = Dense(n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS)
        layer_1.build(input_shape=(self.N_INPUTS,), seed=None)

        xp.random.seed(777)
        layer_2 = Dense(n_inputs=self.N_INPUTS, n_neurons=self.N_NEURONS)
        layer_2.build(input_shape=(self.N_INPUTS,), seed=None)

        self.xp.testing.assert_array_equal(
            layer_1.weights,
            layer_2.weights,
            err_msg="build(seed=None) failed to follow global backend seeding."
        )

    def test_backward_without_input_grad_keeps_parameter_grads(self):
        """needs_input_grad=False drops dX only; dW/dB must match a normal backward."""
        dvalues = self.xp.random.randn(self.BATCH_SIZE, self.N_NEURONS).astype(self.xp.float32)

        self.layer.forward(self.test_inputs, training=True)
        self.assertIsNotNone(self.layer.backward(dvalues))
        expected_dw, expected_db = self.layer.dweights.copy(), self.layer.dbiases.copy()

        self.layer.needs_input_grad = False
        self.layer.forward(self.test_inputs, training=True)
        self.assertIsNone(self.layer.backward(dvalues))
        self.assertIsNone(self.layer.dinputs)
        self.xp.testing.assert_array_equal(self.layer.dweights, expected_dw)
        self.xp.testing.assert_array_equal(self.layer.dbiases, expected_db)

base_case.register_test_suites(globals(), TestDenseLayer)