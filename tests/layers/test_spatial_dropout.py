import aether.config as config
import tests.base_case as base_case

from aether.layers.dropout import SpatialDropout


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


base_case.register_test_suites(globals(), TestSpatialDropout)