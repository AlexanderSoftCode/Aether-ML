import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.layers.pooling import AvgPool2d


class TestAvgPool2d(base_case.AetherBaseLayerTestCase):
    FILTER_SIZE = (2, 2)
    STRIDE = (2, 2)
    PADDING = 'valid'        
    def setUp(self):
        super().setUp()

        # Grab extra utility imports for specific backend if needed
        self.as_strided = config.get_stride_utility(self.xp)
        self.layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(28, 28, 1), 
            filter_size=self.FILTER_SIZE, stride=self.STRIDE, padding=self.PADDING,
        )
        self.test_images = self.xp.random.randn(2, 28, 28, 1)

    def test_forward_average_valid_shape(self):
        """Verify output dimensions for average pooling with 'valid' padding."""
        output = self.layer.forward(self.test_images, training=True)
        self.assertEqual(output.shape, (2, 14, 14, 1))

    def test_forward_average_same_shape(self):
        """Verify output dimensions for average pooling with 'same' padding
        on a non-evenly-divisible input."""
        layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(7, 7, 3), 
            filter_size=self.FILTER_SIZE, stride=self.STRIDE, padding="same",
        )
        inputs = self.xp.random.randn(2, 7, 7, 3)
        output = layer.forward(inputs, training=True)
        self.assertEqual(output.shape, (2, 4, 4, 3))

    def test_forward_average_known_values(self):
        """Verify average pooling computes the correct mean of each window."""
        inputs = self.xp.arange(16, dtype=self.xp.float32).reshape(1, 4, 4, 1)
        expected = self.xp.array(
            [[[[2.5], [4.5]],
              [[10.5], [12.5]]]], dtype=self.xp.float32
        )
        output = self.layer.forward(inputs, training=True)
        self.xp.testing.assert_array_almost_equal(output, expected, decimal=5)

    # ---- Backward pass Average Pooling Non-Overlapping Windows

    def test_backward_average_non_overlapping_valid(self):
        output = self.layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(*output.shape)
        dinputs = self.layer.backward(dvalues)
        self.assertEqual(dinputs.shape, self.test_images.shape)

    def test_backward_average_non_overlapping_same(self):
        # Non-divisible spatial size so 'same' padding is non-trivial.
        inputs = self.xp.random.randn(2, 7, 7, 1)
        output = self.layer.forward(inputs, training=True)
        dvalues = self.xp.random.randn(*output.shape)
        dinputs = self.layer.backward(dvalues)
        self.assertEqual(dinputs.shape, inputs.shape)

    def test_backward_average_even_gradient_distribution(self):
        """Each element inside a non-overlapping average-pooling window should
        receive an equal share (dvalue / window_size) of the upstream gradient."""
        layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(4, 4, 1), 
            filter_size=(2, 2), stride=(2, 2), padding='valid',
        )
        inputs = self.xp.arange(16, dtype=self.xp.float32).reshape(1, 4, 4, 1)
        layer.forward(inputs, training=True)

        dvalues = self.xp.array(
            [[[[4.], [8.]],
              [[12.], [16.]]]], dtype=self.xp.float32
        )
        dinputs = layer.backward(dvalues)

        expected = self.xp.array(
            [[[[1.], [1.], [2.], [2.]],
              [[1.], [1.], [2.], [2.]],
              [[3.], [3.], [4.], [4.]],
              [[3.], [3.], [4.], [4.]]]], dtype=self.xp.float32
        )
        self.xp.testing.assert_array_almost_equal(dinputs, expected, decimal=5)

    # ---- Backward pass Average Pooling Overlapping Windows

    def test_backward_average_overlapping_valid(self):
        layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(28, 28, 1), 
            filter_size=(3, 3), stride=(3, 3), padding=self.PADDING,
        )
        output = layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(*output.shape)
        dinputs = layer.backward(dvalues)
        self.assertEqual(dinputs.shape, self.test_images.shape)

    def test_backward_average_overlapping_same(self):
        layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(28, 28, 1), 
            filter_size=(3, 3), stride=(3, 3), padding='same',
        )
        output = layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(*output.shape)
        dinputs = layer.backward(dvalues)
        self.assertEqual(dinputs.shape, self.test_images.shape)

    def test_forward_asymmetric_filter_and_stride(self):
        """Verify output dimensions and correctness when height and width 
        filters/strides are asymmetric (e.g., filter_size=(3, 2), stride=(2, 1))."""
        layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(7, 5, 1), 
            filter_size=(3, 2), stride=(2, 1), padding='valid',
        )
        inputs = self.xp.arange(1, 36, dtype=self.xp.float32).reshape(1, 7, 5, 1)
        # Output height: (7 - 3) // 2 + 1 = 3
        # Output width:  (5 - 2) // 1 + 1 = 4
        output = layer.forward(inputs, training=True)
        self.assertEqual(output.shape, (1, 3, 4, 1))

    # ---- Backend Parity Regression

    def test_numpy_cupy_parity(self):
        """Verify that CPU (NumPy) and GPU (CuPy) backends produce identical
        numerical results for identical inputs."""
        if self.backend_name != 'cupy':
            self.skipTest("Parity test only executes on GPU (CuPy) backend compared against NumPy.")

        # Generate deterministic CPU data
        np_rng = np.random.RandomState(42)
        raw_input = np_rng.randn(2, 6, 6, 2).astype(np.float32)

        config.set_backend('numpy')
        cpu_layer = AvgPool2d(filter_size=(2, 2), stride=(2, 2), padding='valid')
        cpu_out = cpu_layer.forward(raw_input, training=True)

        config.set_backend('cupy')
        gpu_layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(6, 6, 2), 
            filter_size=(2, 2), stride=(2, 2), padding='valid',
        )
        gpu_out = gpu_layer.forward(self.xp.array(raw_input), training=True)

        self.xp.testing.assert_array_almost_equal(gpu_out, self.xp.array(cpu_out), decimal=5)

    # ---- Data Type Preservation

    def test_forward_backward_dtype_preservation(self):
        """Verify that forward and backward operations preserve input data precision."""
        inputs = self.xp.ones((1, 4, 4, 1), dtype=self.xp.float32)
        output = self.layer.forward(inputs, training=True)
        self.assertEqual(output.dtype, self.xp.float32)

        dvalues = self.xp.ones_like(output, dtype=self.xp.float32)
        dinputs = self.layer.backward(dvalues)
        self.assertEqual(dinputs.dtype, self.xp.float32)

    # ---- Boundary & Negative Value Mechanics

    def test_forward_zero_and_negative_inputs(self):
        """Verify that average pooling correctly preserves signed sums and zeros."""
        inputs = self.xp.array([[[[-4.0], [-2.0]], 
                                 [[ 0.0], [ 2.0]]]], dtype=self.xp.float32)
        # Window mean: (-4.0 - 2.0 + 0.0 + 2.0) / 4 = -1.0
        expected = self.xp.array([[[[-1.0]]]], dtype=self.xp.float32)
        
        output = self.layer.forward(inputs, training=True)
        self.xp.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_same_padding_boundary_scaling(self):
        """Verify boundary pooling behavior under 'same' padding on non-divisible inputs."""
        layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(3, 3, 1), 
            filter_size=(2, 2), stride=(2, 2), padding='same',
        )
        inputs = self.xp.ones((1, 3, 3, 1), dtype=self.xp.float32)
        output = layer.forward(inputs, training=True)
        
        self.assertAlmostEqual(float(output[0, 1, 1, 0]), 0.25, places=5)

    def test_backward_average_overlapping_gradient_accumulation(self):
        """When average-pooling windows overlap, an input cell shared by
        multiple windows should accumulate a contribution from each one."""
        layer = self.make_built_layer(
            AvgPool2d, 
            input_shape=(3, 3, 1), 
            filter_size=(2, 2), stride=(1, 1), padding=self.PADDING,
        )
        inputs = self.xp.random.randn(1, 3, 3, 1)
        layer.forward(inputs, training=True)

        dvalues = self.xp.ones((1, 2, 2, 1), dtype=self.xp.float32)
        dinputs = layer.backward(dvalues)

        expected = self.xp.array(
            [[[[0.25], [0.5], [0.25]],
              [[0.5], [1.0], [0.5]],
              [[0.25], [0.5], [0.25]]]], dtype=self.xp.float32
        )
        self.xp.testing.assert_array_almost_equal(dinputs, expected, decimal=5)
        # Gradient mass is conserved: each window's dvalue is fully
        # redistributed among its own cells, so summing every
        # (possibly overlapping) contribution must equal sum(dvalues).
        self.assertAlmostEqual(
            float(self.xp.sum(dinputs)), float(self.xp.sum(dvalues)), places=5
        )

    def test_backward_average_numerical_gradient(self):
        """Compare analytical gradient against 
        finite-difference approximation for average pooling."""
        epsilon = 1e-2
        fixed_input = self.xp.random.randn(2, 4, 4, 1).astype(dtype=self.xp.float32)
        output = self.layer.forward(fixed_input, training=True)
        dvalues = self.xp.ones_like(output)

        analytical_dinputs = self.layer.backward(dvalues).copy()

        numerical_dinputs = self.xp.zeros_like(fixed_input)

        S, H, W, C = fixed_input.shape
        for s in range(S):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        fixed_input[s, h, w, c] += epsilon
                        loss_plus = self.xp.sum(self.layer.forward(fixed_input, training=False))

                        fixed_input[s, h, w, c] -= 2 * epsilon
                        loss_minus = self.xp.sum(self.layer.forward(fixed_input, training=False))

                        fixed_input[s, h, w, c] += epsilon

                        numerical_dinputs[s, h, w, c] = (loss_plus - loss_minus) / (2 * epsilon)

        self.xp.testing.assert_array_almost_equal(
            analytical_dinputs, numerical_dinputs, decimal=3
        )


base_case.register_test_suites(globals(), TestAvgPool2d)