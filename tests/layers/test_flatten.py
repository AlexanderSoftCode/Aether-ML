import aether.config as config
import tests.base_case as base_case
from aether.layers.linear import Flatten


class TestFlattenLayer(base_case.AetherBaseLayerTestCase):
    INPUT_SHAPE = (4, 7, 7, 3)  # (batch_size, height, width, channels)
    FLATTENED_DIM = 7 * 7 * 3    # 147

    def setUp(self):
        super().setUp()
        self.layer = self.make_built_layer(Flatten, input_shape=self.INPUT_SHAPE[1:])
        
        self.test_inputs = self.xp.arange(
            4 * 7 * 7 * 3, dtype=self.xp.float32
        ).reshape(self.INPUT_SHAPE)

    def test_flatten_forward_shape(self):
        """Verify output is flattened to 2D (batch_size, total_features)."""
        output = self.layer.forward(self.test_inputs, training=True)
        expected_shape = (self.INPUT_SHAPE[0], self.FLATTENED_DIM)
        
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(output.flags.c_contiguous)

    def test_flatten_forward_values(self):
        """Verify flattened output maintains contiguous row-major ordering."""
        output = self.layer.forward(self.test_inputs, training=True)
        expected_first_sample = self.test_inputs[0].ravel()
        
        self.assertTrue(
            self.xp.allclose(output[0], expected_first_sample)
        )
        

    def test_flatten_inference_does_not_cache_shape(self):
        """Verify that inputs_shape is NOT stored/cached when training=False."""
        self.assertFalse(hasattr(self.layer, 'inputs_shape'))

        output = self.layer.forward(self.test_inputs, training=False)
        
        expected_shape = (self.INPUT_SHAPE[0], self.FLATTENED_DIM)
        self.assertEqual(output.shape, expected_shape)

        self.assertFalse(
            hasattr(self.layer, 'inputs_shape'),
            "Flatten layer should not store `inputs_shape` when training=False."
        )
    def test_flatten_backward_shape(self):
        """Verify backward pass reshapes gradients to match the original input shape."""
        self.layer.forward(self.test_inputs, training=True)
        
        dvalues = self.xp.ones((self.INPUT_SHAPE[0], self.FLATTENED_DIM), dtype=self.xp.float32)
        dinputs = self.layer.backward(dvalues)

        self.assertEqual(dinputs.shape, self.INPUT_SHAPE)

    def test_flatten_gradient_integrity(self):
        """Verify backward pass preserves exact gradient values without distortion."""
        self.layer.forward(self.test_inputs, training=True)
        
        dvalues = self.xp.arange(
            self.INPUT_SHAPE[0] * self.FLATTENED_DIM, dtype=self.xp.float32
        ).reshape(self.INPUT_SHAPE[0], self.FLATTENED_DIM)

        dinputs = self.layer.backward(dvalues)
        
        reconstructed_dvalues = dinputs.reshape(self.INPUT_SHAPE[0], -1)
        self.assertTrue(self.xp.allclose(reconstructed_dvalues, dvalues))


base_case.register_test_suites(globals(), TestFlattenLayer)