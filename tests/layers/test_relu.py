import aether.config as config
import tests.base_case as base_case
from aether.layers.activations import ReLU


class TestReLU(base_case.AetherBaseLayerTestCase):
    def setUp(self):
        super().setUp()
        self.layer = self.make_built_layer(
            ReLU, input_shape=(3,)
        )

    def test_forward_pass(self):
        """Verify that the layer correctly outputs max(0, x)."""

        # Setup input with negative, zero, and positive values
        inputs = self.xp.array([
            [-3.0, -1.0, 0.0],
            [0.5, 2.0, 5.0]
        ], dtype=self.xp.float32)

        expected_output = self.xp.array([
            [0.0, 0.0, 0.0],
            [0.5, 2.0, 5.0]
        ], dtype=self.xp.float32)

        # Execute forward pass using the instance layer
        self.layer.forward(inputs, training=False)

        actual_output = self.layer.output
        self.xp.testing.assert_array_almost_equal(
            actual_output,
            expected_output,
            decimal=4,
            err_msg="Forward pass failed: did not clamp negative values correctly."
        )

    def test_backward_pass(self):
        """Verify that gradients only flow through positive input paths."""

        # Inputs to dictate the active/inactive mask
        inputs = self.xp.array([
            [-2.0, 0.0, 1.0, 3.0]
        ], dtype=self.xp.float32)

        # Upstream gradients received from the next layer
        dvalues = self.xp.array([
            [0.5, 0.5, 0.5, 0.5]
        ], dtype=self.xp.float32)

        # Expected: 0 gradient for inputs <= 0, and passed-through gradient for inputs > 0
        expected_dinputs = self.xp.array([
            [0.0, 0.0, 0.5, 0.5]
        ], dtype=self.xp.float32)

        self.layer.forward(inputs, training=False)
        self.layer.backward(dvalues)

        actual_dinputs = self.layer.dinputs

        self.xp.testing.assert_array_almost_equal(
            actual_dinputs,
            expected_dinputs,
            decimal=4,
            err_msg="Backward pass failed: gradient routing mismatch."
        )

    def test_forward_does_not_mutate_input(self):
        """Forward pass should not alter incoming inputs."""
        layer = self.make_built_layer(ReLU, input_shape=(4,))
        inputs = self.xp.array([
            [-2.0, 0.0, 1.0, 3.0]
        ], dtype=self.xp.float32)
        original_inputs = inputs.copy()

        layer.forward(inputs, training=False)
        self.xp.testing.assert_array_equal(inputs, original_inputs)

    def test_backward_does_not_mutate_dvalues(self):
        """Backward pass should not alter incoming dvalues."""
        layer = self.make_built_layer(ReLU, input_shape=(4,))

        inputs = self.xp.array([
            [-2.0, 0.0, 1.0, 3.0]
        ], dtype=self.xp.float32)

        layer.forward(inputs, training=True)

        dvalues = self.xp.array([
            [0.5, 1.5, -2.0, 3.0]
        ], dtype=self.xp.float32)
        original_dvalues = dvalues.copy()

        layer.backward(dvalues)

        self.xp.testing.assert_array_equal(dvalues, original_dvalues)


base_case.register_test_suites(globals(), TestReLU)