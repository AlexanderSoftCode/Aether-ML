import tests.base_case as base_case
from aether.layers.conv import Conv2d
from tests.layers.conv2d.base import ConvLayerTestBase


class TestConvInit(ConvLayerTestBase):
    """Parameter initialisation and build() seeding."""

    def test_parameters_initialization(self):
        """Compare filter weights and filter bias shapes"""
        self.assertEqual(self.layer.filter_weights.shape, (3, 3, 1, 4))
        self.assertEqual(self.layer.biases.shape, (4,))

    def test_same_seed_produces_identical_weights(self):
        """Two independently instantiated layers built with the exact same seed
        must initialize with identical weight and bias values."""
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )

        seed_val = 123
        layer_1.build(input_shape = self.CONV_SHAPE[1:], seed=seed_val)
        layer_2.build(input_shape = self.CONV_SHAPE[1:], seed=seed_val)

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
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )

        layer_1.build(input_shape=self.CONV_SHAPE[1:], seed=123)
        layer_2.build(input_shape=self.CONV_SHAPE[1:], seed=456)

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
        throwaway_layer = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        throwaway_layer.build(input_shape = self.CONV_SHAPE[1:], seed=123)

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
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_1.build(input_shape = self.CONV_SHAPE[1:], seed=None)

        xp.random.seed(777)
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2.build(input_shape = self.CONV_SHAPE[1:], seed=None)

        self.xp.testing.assert_array_equal(
            layer_1.weights,
            layer_2.weights,
            err_msg="build(seed=None) failed to follow global backend seeding."
        )


base_case.register_test_suites(globals(), TestConvInit)
