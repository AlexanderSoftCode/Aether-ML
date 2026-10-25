import numpy as np

import tests.base_case as base_case
from aether.layers.conv import Conv2d
from tests.layers.conv2d.base import ConvLayerTestBase


class TestConvForward(ConvLayerTestBase):
    """Forward shapes, padding and stride geometry, inference caches."""

    def test_conv_forward_shape(self):
        """Verify output dimensions based on padding and stride"""
        output = self.layer.forward(self.test_images, training=True)
        # Given padding = 1, kernel = 3, 28x28 remainds 28x28 (same)
        self.assertEqual(output.shape, (2, 28, 28, 4))

    def test_conv_same_padding_stride(self):
        # 28x28 input, 3x3 filter, stride=(2,2), padding='same'
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=(2,2),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training=True)

        # Calculation: P=1 -> int((28 + 2 - 3)/2 + 1) = 14
        self.assertEqual(output.shape, (2, 14, 14, 4))

    def test_conv_valid_padding_stride(self):
        # 28x28 input, 3x3 filter, stride = (2,2), valid padding
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=(2,2),
            padding="valid"
        )
        output = layer.forward(self.test_images, training=True)
        # (28 - 3) / 2 + 1 = 13.5 -> floor to 13
        self.assertEqual(output.shape, (2, 13, 13, 4))

    def test_conv_zero_input(self):
        "Given an input of all zeros, show an output of all zeros (plus biases)"

        zero_input = self.xp.zeros((1, 28, 28, 1))

        # Expected: Every pixel in the output should equal the bias for that filter
        # We reshape biases to (1, 1, 1, num_filters) to match output shape

        output = self.layer.forward(zero_input, training = False)
        expected_output = self.xp.broadcast_to(self.layer.biases, output.shape)

        self.xp.testing.assert_array_almost_equal(output, expected_output)

    def test_conv_stride_length(self):

        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=(2,1),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training = False)

        self.assertEqual(output.shape, (2, 14, 28, 4))

    def test_conv_stride_width(self):
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=(1,2),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training=False)
        self.assertEqual(output.shape, (2, 28, 14, 4))

    def test_conv_multichannel_input(self):
        """Verify forward pass with multi-channel input (e.g., RGB with C_in=3)"""
        layer = self.make_built_layer(
            Conv2d,
            input_shape = (28, 28, 3),
            in_channels=3,
            out_channels=8,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        rgb_images = self.xp.random.randn(2, 28, 28, 3)
        output = layer.forward(rgb_images, training=False)
        self.assertEqual(output.shape, (2, 28, 28, 8))

    def test_conv_variable_batch_sizes(self):
        """Verify shape cache handles dynamic batch sizes without re-compilation or shape errors"""
        inputs_b4 = self.xp.random.randn(4, 16, 16, self.IN_CHANNELS)
        out_b4 = self.layer.forward(inputs_b4, training=False)
        self.assertEqual(out_b4.shape, (4, 16, 16, self.OUT_CHANNELS))

        inputs_b1 = self.xp.random.randn(1, 16, 16, self.IN_CHANNELS)
        out_b1 = self.layer.forward(inputs_b1, training=False)
        self.assertEqual(out_b1.shape, (1, 16, 16, self.OUT_CHANNELS))

    def test_conv_non_square_filter(self):
        """Verify rectangular filter geometries such as (5, 3)"""
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=(5, 3),
            stride=(1, 1),
            padding="same"
        )
        output = layer.forward(self.test_images, training=False)
        self.assertEqual(output.shape, (2, 28, 28, self.OUT_CHANNELS))

    def test_same_padding_build_matches_forward_shape(self):
        """Regression: build() and forward() once derived 'same' geometry separately
        and disagreed for even filters (8x8, 2x2 -> build 8x8, forward 7x7)."""
        for spatial, filter_size, stride in self.SAME_CASES:
            with self.subTest(spatial=spatial, filter_size=filter_size, stride=stride):
                layer = self._make_same_layer(spatial, filter_size, stride)
                out = layer.forward(
                    self.xp.zeros((2, *spatial, 2), dtype=self.xp.float32), training=False
                )
                expected = tuple(-(-n // s) for n, s in zip(spatial, stride))
                self.assertEqual(layer.output_shape, (*expected, 3))
                self.assertEqual(out.shape[1:], layer.output_shape)

    def test_same_padding_forward_ground_truth(self):
        """Forward against the definition, with TF's pad_top = pad_total // 2."""
        for spatial, filter_size, stride in self.SAME_CASES:
            with self.subTest(spatial=spatial, filter_size=filter_size, stride=stride):
                layer = self._make_same_layer(spatial, filter_size, stride)
                x = self.xp.random.randn(2, *spatial, 2).astype(self.xp.float32)
                out = self._to_numpy(layer._forward_fallback(x, training=False))

                xn, w = self._to_numpy(x), self._to_numpy(layer.filter_weights)
                (H, W), (fH, fW), (sH, sW) = spatial, filter_size, stride
                H_out, W_out = out.shape[1:3]
                pad_h = max((H_out - 1) * sH + fH - H, 0) // 2
                pad_w = max((W_out - 1) * sW + fW - W, 0) // 2
                padded = np.pad(xn, ((0, 0), (pad_h, fH), (pad_w, fW), (0, 0)))
                truth = np.zeros_like(out)
                for h in range(H_out):
                    for v in range(W_out):
                        patch = padded[:, h * sH:h * sH + fH, v * sW:v * sW + fW, :]
                        truth[:, h, v, :] = np.tensordot(patch, w, axes=([1, 2, 3], [0, 1, 2]))
                np.testing.assert_allclose(out, truth, rtol=1e-5, atol=1e-5)

    def test_inference_forward_drops_backward_caches(self):
        """Inference keeps no activation-sized caches, and backward then refuses."""
        self.layer.forward(self.test_images, training=True)
        self.assertIsNotNone(self.layer.inputs)
        self.layer.forward(self.test_images, training=False)
        self.assertIsNone(self.layer.inputs)
        self.assertIsNone(getattr(self.layer, "padded_inputs", None))
        with self.assertRaises(RuntimeError):
            self.layer.backward(self.xp.ones((2, 28, 28, self.OUT_CHANNELS)))


base_case.register_test_suites(globals(), TestConvForward)
