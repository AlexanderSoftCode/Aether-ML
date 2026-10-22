import aether.config as config
import tests.base_case as base_case
from aether.layers.activations import LeakyReLU


class TestLeakyReLU(base_case.AetherBaseLayerTestCase):
    ALPHA = 0.1

    def setUp(self):
        super().setUp()
        self.layer = self.make_built_layer(
            LeakyReLU, input_shape=(3,), alpha=self.ALPHA
        )

    def test_forward_pass(self):
        """Verify output values for leaky_ReLU"""            
        # Setup input with negative, zero, and positive values
        inputs = self.xp.array([
            [-3.0, -1.0, 0.0], 
            [0.5, 2.0, 5.0]
        ], dtype=self.xp.float32)

        expected_output = self.xp.array([
            [-0.3, -0.1, 0.0],
            [0.5, 2.0, 5.0]
        ], dtype =self.xp.float32)

        actual_output = self.layer.forward(inputs, training=False)

        self.xp.testing.assert_array_almost_equal(
            actual_output,
            expected_output,
            decimal=4,
            err_msg="Forward pass failed: did not clamp negative values correctly"
        )

    def test_analytical_gradients_backward_pass(self): 
        """Verify that gradients flow through input paths ."""

        # inputs to dictate the active/inactive mask
        inputs = self.xp.array([
            [-2.0, 0.0, 1.0, 3.0],
            [-4.0, 5.0, 10.0, -15.0]
        ], dtype=self.xp.float32)

        # Upstream gradients received from the next layaer

        dvalues = self.xp.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0]
        ], dtype=self.xp.float32)

        # Expected x for x>0, expected alpha*x for x<0
        expected_dinputs = self.xp.array([
            [0.05, 0.05, 0.5, 0.5],
            [0.1, 1.0, 1.0, 0.1]
        ], dtype=self.xp.float32)

        self.layer.forward(inputs, training=False)
        self.layer.backward(dvalues)

        actual_dinputs = self.layer.dinputs 

        self.xp.testing.assert_array_almost_equal(
            actual_dinputs,
            expected_dinputs,
            decimal=4,
            err_msg="Backward pass failed: gradient routing mismatch"
        )

    def test_backend_pointer_swap(self):
        """Verify that LeakyReLU dynamically binds the correct
        operational methods based on input context."""

        current_forward_name = self.layer.forward.__name__
        current_backward_name = self.layer.backward.__name__

        if self.backend_name == 'cupy':
            # Assert that the pointers have swapped to the faster fused GPU variants
            self.assertEqual(
                current_forward_name, "_forward_gpu",
                msg="CuPy backend failed to bind the optimized GPU forward pointer."
            )
            self.assertEqual(
                current_backward_name, "_backward_gpu",
                msg="CuPy backend failed to bind the optimized GPU backward pointer."
            )
        else:
            # Assert that the pointers default cleanly to standard host CPU branches
            self.assertEqual(
                current_forward_name, "_forward_fallback",
                msg="NumPy backend failed to bind the host CPU forward pointer."
            )
            self.assertEqual(
                current_backward_name, "_backward_fallback",
                msg="NumPy backend failed to bind the host CPU backward pointer."
            )
    
    def test_gradient_memory_safety(self):
        """ Verify backward pass does not alter the incoming dvalues tensor in-place."""
        layer = self.make_built_layer(LeakyReLU, input_shape=(4,), alpha=self.ALPHA)
        inputs = self.xp.array([
            [-2.0, 0.0, 1.0, 3.0],
            [ 0.5, -1.0, 2.0, -0.5]
        ], dtype=self.xp.float32)

        dvalues = self.xp.array([
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0]
        ], dtype=self.xp.float32)

        desired_dvalues = dvalues.copy()
        layer.forward(inputs, training=False)
        layer.backward(dvalues)

        self.xp.testing.assert_array_equal(
            desired_dvalues,
            dvalues,
            err_msg="dvalues is being mutated in-place during the backward pass."
        )

    def test_custom_alpha_scaling(self):
        """Verify hyperparameter integrity"""
        custom_alpha = 0.25
        custom_layer = self.make_built_layer(LeakyReLU, input_shape=(2,), alpha=custom_alpha)
        inputs = self.xp.array([[-2.0, -4.0]], dtype=self.xp.float32)
        expected_output = self.xp.array([[-0.5, -1.0]], dtype=self.xp.float32)

        custom_layer.forward(inputs, training=False)

        self.xp.testing.assert_array_almost_equal(
            custom_layer.output,
            expected_output,
            decimal=4,
            err_msg=f"Hyperparameter initialization mismatch: Custom alpha scaling failed to apply value {custom_alpha} properly."
        )

base_case.register_test_suites(globals(), TestLeakyReLU)