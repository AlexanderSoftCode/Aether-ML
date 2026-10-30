import tests.base_case as base_case
from aether.layers.activations import SoftMax


class TestSoftmax(base_case.AetherBaseLayerTestCase):
    def setUp(self):
        super().setUp()
        self.layer = self.make_built_layer(SoftMax, input_shape=(3,))

    def test_forward_pass_probabilities(self):
        """Verify output probabilities sum to 1.0 along the feature axis."""
        inputs = self.xp.array([
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 1.0]
        ], dtype=self.xp.float32)

        output = self.layer.forward(inputs, training=False)

        # Probabilities along axis -1 must sum to 1.0
        sums = self.xp.sum(output, axis=-1)
        expected_sums = self.xp.ones_like(sums)

        self.xp.testing.assert_allclose(
            sums,
            expected_sums,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Forward pass failed: probabilities do not sum to 1.0."
        )

    def test_forward_numerical_stability(self):
        """Verify layer handles large input values without numerical overflow."""
        inputs = self.xp.array([
            [1000.0, 1001.0, 1002.0],
            [-1000.0, -1001.0, -1002.0]
        ], dtype=self.xp.float32)

        output = self.layer.forward(inputs, training=False)

        self.assertFalse(
            bool(self.xp.any(self.xp.isnan(output))),
            "Forward pass produced NaN values due to numerical instability."
        )
        self.assertFalse(
            bool(self.xp.any(self.xp.isinf(output))),
            "Forward pass produced Inf values due to numerical instability."
        )

    def test_backward_pass_gradients(self):
        """Verify the full Jacobian backward pass matches the analytical gradient."""
        inputs = self.xp.array([
            [1.0, 2.0, 3.0],
            [0.5, -0.5, 1.5]
        ], dtype=self.xp.float32)

        dvalues = self.xp.array([
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4]
        ], dtype=self.xp.float32)

        self.layer.forward(inputs, training=True)
        self.layer.backward(dvalues)

        actual_dinputs = self.layer.dinputs

        # Analytical Jacobian: dinputs = output * (dvalues - sum(dvalues * output, axis=-1, keepdims=True))
        output = self.layer.output
        dot_product = self.xp.sum(dvalues * output, axis=-1, keepdims=True)
        expected_dinputs = output * (dvalues - dot_product)

        self.xp.testing.assert_allclose(
            actual_dinputs,
            expected_dinputs,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Backward pass failed: analytical gradient mismatch."
        )

    def test_forward_does_not_mutate_input(self):
        """Forward pass should not alter incoming inputs."""
        inputs = self.xp.array([
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 1.0]
        ], dtype=self.xp.float32)
        original_inputs = inputs.copy()

        self.layer.forward(inputs, training=False)
        self.xp.testing.assert_array_equal(inputs, original_inputs)

    def test_backward_does_not_mutate_dvalues(self):
        """Backward pass should not alter incoming dvalues."""
        inputs = self.xp.array([
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 1.0]
        ], dtype=self.xp.float32)

        self.layer.forward(inputs, training=True)

        dvalues = self.xp.array([
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4]
        ], dtype=self.xp.float32)
        original_dvalues = dvalues.copy()

        self.layer.backward(dvalues)
        self.xp.testing.assert_array_equal(dvalues, original_dvalues)


base_case.register_test_suites(globals(), TestSoftmax)