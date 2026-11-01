import tests.base_case as base_case
from aether.layers.dropout import SpatialDropout
from tests.layers.philox_reference import kept_elements


class TestSpatialDropout(base_case.AetherBaseLayerTestCase):
    DEFAULT_RATE = 0.5
    FIXED_SEED = 12345
    INPUT_SHAPE = (28, 28, 8)

    def setUp(self):
        super().setUp()

        self.layer = self.make_built_layer(
            SpatialDropout, input_shape=(self.INPUT_SHAPE), seed=self.FIXED_SEED, rate=self.DEFAULT_RATE
        )
        # Skip certain GPU tests using the flag when running on np backend
        self.uses_gpu_kernel = (self.layer.forward.__name__ == '_forward_gpu')

    def test_keep_rate_computed_from_rate(self):
        rate = 0.35
        layer = self.make_built_layer(SpatialDropout, input_shape=self.INPUT_SHAPE, seed=self.FIXED_SEED, rate=rate)
        self.assertAlmostEqual(layer.keep_rate, 1-rate, places=7)
    
    def test_forward_output_shape_matches_input_Id(self):
        inputs = self.xp.ones((2, 28, 28, 8), dtype=self.xp.float32)
        output = self.layer.forward(inputs, training=True)

        self.assertEqual(output.shape, inputs.shape)

    def test_forward_eval_mode_returns_input_unchanged(self):
        # 2 * 4 * 4 * 3 = 96 total elements
        layer = self.make_built_layer(
            SpatialDropout,
            input_shape=(4, 4, 3),
            seed=self.FIXED_SEED,
            rate=self.DEFAULT_RATE,
        )
        inputs = self.xp.arange(96, dtype=self.xp.float32).reshape(2, 4, 4, 3)
        output = layer.forward(inputs, training=False)

        # Eval mode is the identity: no scaling, no copy, no allocation.
        self.assertIs(output, inputs)

    def test_backward_after_eval_forward_raises(self):
        layer = self.make_built_layer(
            SpatialDropout,
            input_shape=(4, 4, 3),
            seed=self.FIXED_SEED,
            rate=self.DEFAULT_RATE,
        )
        inputs = self.xp.arange(96, dtype=self.xp.float32).reshape(2, 4, 4, 3)
        layer.forward(inputs, training=False)

        with self.assertRaises(RuntimeError):
            layer.backward(self.xp.ones_like(inputs))

    def test_forward_training_zero_rate_is_identity(self):
        layer = self.make_built_layer(SpatialDropout, input_shape = (10, 10, 5), seed=self.FIXED_SEED, rate=0.0)
        # 2 * 10 * 10 * 5 = 1000 total elements
        inputs = self.xp.linspace(0.1, 5.0, 1000, dtype=self.xp.float32).reshape(2, 10, 10, 5)
        output = layer.forward(inputs, training=True)
        self.xp.testing.assert_allclose(inputs, output, rtol=1e-4)

    def test_forward_scales_kept_units_by_inverse_keep_rate(self):
        rate = 0.3
        keep_rate = 1 - rate
        layer = self.make_built_layer(SpatialDropout, input_shape= (10, 10, 6), seed=self.FIXED_SEED, rate=rate)
        tensor_shape = (3, 10, 10, 6)
        inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)

        if not self.uses_gpu_kernel:
            self.xp.random.seed(self.FIXED_SEED)

        output = layer.forward(inputs, training=True)
        kept = output[output!=0]
        self.assertTrue(kept.size > 0)
        self.assertTrue(bool(self.xp.allclose(kept, 1.0 / keep_rate, rtol = 1e-4)))

    def test_forward_drops_approximately_expected_fraction(self):
        rate = 0.4
        # Pass the fixed seed directly so both CPU and GPU kernels generate deterministic masks
        layer = self.make_built_layer(SpatialDropout, input_shape=(2, 2, 1000), rate=rate, seed=self.FIXED_SEED)
        
        tensor_shape = (50, 2, 2, 1000)
        inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)

        output = layer.forward(inputs, training=True)
        dropped_fraction = float(self.xp.mean((output == 0).astype(self.xp.float32)))
        self.assertAlmostEqual(dropped_fraction, rate, delta=0.08)

    def test_backward_uses_same_mask_and_scale_as_forward(self):
        layer = self.make_built_layer(SpatialDropout, input_shape=(4, 4, 20), rate=0.5)
        tensor_shape = (1, 4, 4, 20)
        inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)
        forward_out = layer.forward(inputs, training=True)
        dvalues = self.xp.ones(tensor_shape, dtype=self.xp.float32)
        dinputs = layer.backward(dvalues)

        self.assertTrue(bool(self.xp.allclose(forward_out, dinputs)))

    def test_repeated_forward_in_same_step_reuses_mask(self):
        tensor_shape = (2, 10, 10, 25)
        layer = self.make_built_layer(
            SpatialDropout,
            input_shape=(10, 10, 25),
            seed=self.FIXED_SEED,
            rate=self.DEFAULT_RATE,
        )
        inputs = self.xp.ones(tensor_shape, dtype=self.xp.float32)

        first = layer.forward(inputs, training=True).copy()
        second = layer.forward(inputs, training=True)

        self.assertTrue(bool(self.xp.all(first == second)))

    def test_forward_across_steps_uses_different_masks(self):
        tensor_shape = (2, 10, 10, 25)
        layer = self.make_built_layer(
            SpatialDropout,
            input_shape=(10, 10, 25),
            seed=self.FIXED_SEED,
            rate=self.DEFAULT_RATE,
        )
        inputs = self.xp.ones(tensor_shape, dtype=self.xp.float32)

        first = layer.forward(inputs, training=True).copy()
        layer._clock.advance()
        second = layer.forward(inputs, training=True)

        self.assertFalse(bool(self.xp.all(first == second)))

    # ---- per-sample masks ---------------------------------------------

    def _channel_masks(self, shape=(8, 4, 4, 64)):
        layer = self.make_built_layer(
            SpatialDropout, input_shape=shape[1:], seed=self.FIXED_SEED, rate=self.DEFAULT_RATE
        )
        output = layer.forward(self.xp.ones(shape, dtype=self.xp.float32), training=True)
        return output != 0

    def test_samples_draw_independent_channel_masks(self):
        """Regression: one (1, 1, 1, C) mask was once shared by the whole batch."""
        kept = self._channel_masks()[:, 0, 0, :]
        self.assertFalse(bool(self.xp.all(kept == kept[0])))

    def test_mask_is_constant_over_each_feature_map(self):
        """Within one sample, a channel is kept or dropped as a whole."""
        kept = self._channel_masks()
        self.assertTrue(bool(self.xp.all(kept == kept[:, :1, :1, :])))

    def test_backward_matches_forward_mask_per_sample(self):
        layer = self.make_built_layer(
            SpatialDropout, input_shape=(3, 3, 16), seed=self.FIXED_SEED, rate=self.DEFAULT_RATE
        )
        inputs = self.xp.ones((6, 3, 3, 16), dtype=self.xp.float32)
        forward_out = layer.forward(inputs, training=True)
        dinputs = layer.backward(self.xp.ones_like(inputs))
        self.assertTrue(bool(self.xp.allclose(forward_out, dinputs)))

    # ---- clock / offset bookkeeping -----------------------------------

    def test_forward_does_not_advance_the_clock(self):
        tensor_shape = (2, 5, 5, 25)
        layer = self.make_built_layer(
            SpatialDropout, input_shape=(5, 5, 25), seed=self.FIXED_SEED, rate=0.5
        )
        inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)

        self.assertEqual(layer._clock.value, 0)
        layer.forward(inputs, training=True)
        self.assertEqual(layer._clock.value, 0)
        layer.forward(inputs, training=True)
        self.assertEqual(layer._clock.value, 0)

    def test_active_offset_tracks_the_clock(self):
        tensor_shape = (2, 5, 5, 25)
        layer = self.make_built_layer(
            SpatialDropout, input_shape=(5, 5, 25), seed=self.FIXED_SEED, rate=0.5
        )
        inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)

        layer.forward(inputs, training=True)
        self.assertEqual(layer._active_offset, 0)

        layer._clock.advance()
        layer.forward(inputs, training=True)
        self.assertEqual(layer._active_offset, 1)

    def test_eval_mode_clears_active_offset(self):
        tensor_shape = (2, 5, 5, 25)
        layer = self.make_built_layer(
            SpatialDropout, input_shape=(5, 5, 25), seed=self.FIXED_SEED, rate=0.5
        )
        inputs = self.xp.ones(shape=tensor_shape, dtype=self.xp.float32)

        layer.forward(inputs, training=True)
        layer.forward(inputs, training=False)

        self.assertEqual(layer._clock.value, 0)
        self.assertEqual(layer._active_offset, -1)


    # ---- GPU kernel against the reference ------------------------------------

    def _pinned_layer(self, input_shape, rate):
        """make_built_layer's seed only reaches build(), so pin the mask stream directly."""
        layer = self.make_built_layer(SpatialDropout, input_shape=input_shape, rate=rate)
        layer._seed_key = self.FIXED_SEED
        return layer

    def test_gpu_mask_matches_reference_per_sample_and_channel(self):
        if not self.uses_gpu_kernel:
            self.skipTest("Philox kernel only runs on the GPU path")

        S, C = 9, 13
        layer = self._pinned_layer((3, 3, C), 0.3)
        layer._clock.value = 4
        output = layer.forward(self.xp.ones((S, 3, 3, C), dtype=self.xp.float32), training=True)

        expected = kept_elements(layer._seed_key, 4, S * C, layer._keep_threshold)
        self.assertEqual((output[:, 0, 0, :] != 0).get().ravel().tolist(), expected)

    def test_backward_applies_the_forward_mask_after_the_clock_advances(self):
        layer = self._pinned_layer((2, 2, 16), 0.5)
        shape = (4, 2, 2, 16)
        kept = layer.forward(self.xp.ones(shape, dtype=self.xp.float32), training=True) != 0
        layer._clock.advance()

        dvalues = self.xp.arange(1, 257, dtype=self.xp.float32).reshape(shape)
        dinputs = layer.backward(dvalues)

        # Hand-derived: rate 0.5 scales kept elements by exactly 2.
        self.assertTrue(bool(self.xp.all(dinputs[kept] == dvalues[kept] * 2.0)))
        self.assertTrue(bool(self.xp.all(dinputs[~kept] == 0.0)))

    def test_output_follows_input_dtype(self):
        for dtype in (self.xp.float16, self.xp.float32, self.xp.float64):
            layer = self._pinned_layer((2, 2, 16), 0.5)
            output = layer.forward(self.xp.full((4, 2, 2, 16), 1.5, dtype=dtype), training=True)
            self.assertEqual(output.dtype, dtype)
            self.assertTrue(bool(self.xp.all(output[output != 0] == 3.0)))

    def test_eval_forward_releases_the_gpu_mask(self):
        if not self.uses_gpu_kernel:
            self.skipTest("the cached mask is only released on the GPU path")

        inputs = self.xp.ones((2, 5, 5, 25), dtype=self.xp.float32)
        self.layer.forward(inputs, training=True)
        self.assertEqual(self.layer.channel_mask.shape, (2, 1, 1, 25))

        self.layer.forward(inputs, training=False)
        self.assertIsNone(self.layer.channel_mask)

    def test_dropped_channels_are_zero_even_for_nan_and_inf(self):
        xp = self.xp
        shape = (4, 2, 2, 16)
        layer = self._pinned_layer(shape[1:], 0.5)
        kept = layer.forward(xp.ones(shape, dtype=xp.float32), training=True) != 0
        self.assertTrue(bool(xp.any(kept)) and not bool(xp.all(kept)))

        # Every other channel is NaN, the rest +inf, so both land in kept and dropped.
        bad = xp.full(shape, float('inf'), dtype=xp.float32)
        bad[..., ::2] = float('nan')
        for result in (layer.forward(bad, training=True), layer.backward(bad)):
            self.assertTrue(bool(xp.all(result[~kept] == 0.0)))
            self.assertTrue(bool(xp.all(xp.isnan(result[kept]) == xp.isnan(bad[kept]))))
            self.assertTrue(bool(xp.all(xp.isinf(result[kept]) == xp.isinf(bad[kept]))))

base_case.register_test_suites(globals(), TestSpatialDropout)