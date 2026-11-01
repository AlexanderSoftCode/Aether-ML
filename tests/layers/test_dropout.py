import numpy as np

import aether as ae
import aether.config as config
import tests.base_case as base_case
from aether.custom_kernels import dropout_kernel as gpu_dropout
from aether.layers.dropout import Dropout
from aether.layers.linear import Dense
from aether.model import Model
from tests.layers.philox_reference import kept_elements, philox4x32_10


class TestDropout(base_case.AetherBaseLayerTestCase):
    DEFAULT_RATE = 0.5
    FIXED_SEED = 12345

    def setUp(self):
        super().setUp()

        self.layer = self.make_built_layer(
            Dropout, input_shape=(16,), rate=self.DEFAULT_RATE, seed=self.FIXED_SEED
        )
        self.uses_gpu_kernel = (self.layer.forward.__name__ == '_forward_gpu')
     
    def test_keep_rate_computed_from_rate(self):
        rate = 0.35
        layer = self.make_built_layer(Dropout, input_shape=(16,), rate=rate)
        self.assertAlmostEqual(layer.keep_rate, 1 - rate, places=7)

    def test_forward_output_shape_matches_input_1d(self):
        inputs = self.xp.ones(1000, dtype=self.xp.float32)
        output = self.layer.forward(inputs, training=True)
        self.assertEqual(output.shape, inputs.shape)

    def test_forward_preserves_multidimensional_input_shape(self):
        """Dropout is an elementwise layer, so it should work on inputs
        of any rank (e.g. (batch, features) activations coming out of a
        Dense/Conv layer), not just flat vectors."""
        inputs = self.xp.ones((8, 16), dtype=self.xp.float32)
        output = self.layer.forward(inputs, training=True)
        self.assertEqual(output.shape, inputs.shape)

    def test_forward_eval_mode_returns_input_unchanged(self):
        inputs = self.xp.arange(20, dtype=self.xp.float32).reshape(4, 5)
        output = self.layer.forward(inputs, training=False)
        self.assertTrue(bool(self.xp.all(output == inputs)))


    def test_forward_training_zero_rate_is_identity(self):
        layer = self.make_built_layer(Dropout, input_shape=(500,), rate=0.0)
        inputs = self.xp.linspace(0.1, 5.0, 500, dtype=self.xp.float32)
        output = layer.forward(inputs, training=True)
        self.assertTrue(bool(self.xp.allclose(output, inputs, rtol=1e-5)))

    def test_forward_scales_kept_units_by_inverse_keep_rate(self):
        rate = 0.3
        keep_rate = 1 - rate
        n = 20000
        layer = self.make_built_layer(Dropout, input_shape=(n,), rate=rate)
        
        inputs = self.xp.ones(n, dtype=self.xp.float32)

        if not self.uses_gpu_kernel:
            self.xp.random.seed(self.FIXED_SEED)

        output = layer.forward(inputs, training=True)
        kept = output[output != 0]
        self.assertTrue(kept.size > 0)
        self.assertTrue(bool(self.xp.allclose(kept, 1.0 / keep_rate, rtol=1e-4)))

    def test_forward_drops_approximately_expected_fraction(self):
        rate = 0.4
        n = 100000
        layer = self.make_built_layer(Dropout, input_shape=(n,), rate=rate)

        inputs = self.xp.ones(n, dtype=self.xp.float32)

        if not self.uses_gpu_kernel:
            self.xp.random.seed(self.FIXED_SEED)

        output = layer.forward(inputs, training=True)
        dropped_fraction = float(self.xp.mean((output == 0).astype(self.xp.float32)))
        self.assertAlmostEqual(dropped_fraction, rate, delta=0.02)

    def test_backward_uses_same_mask_and_scale_as_forward(self):
        n = 5000
        layer = self.make_built_layer(Dropout, input_shape=(n,), rate=0.5)
        
        inputs = self.xp.ones(n, dtype=self.xp.float32)

        forward_out = layer.forward(inputs, training=True)
        dvalues = self.xp.ones(n, dtype=self.xp.float32)
        dinputs = layer.backward(dvalues)

        self.assertTrue(bool(self.xp.allclose(forward_out, dinputs)))

    def test_repeated_forward_in_same_step_reuses_mask(self):
        n = 50
        inputs = self.xp.ones(n, dtype=self.xp.float32)
        first = self.layer.forward(inputs, training=True).copy()
        second = self.layer.forward(inputs, training=True)
        self.assertTrue(bool(self.xp.all(first == second)))

    # ---- clock / offset bookkeeping -----------------------------------
    # These reach into _clock and _active_offset deliberately: the contract under
    # test *is* the RNG bookkeeping, so the coupling to internals is intentional.

    def test_forward_does_not_advance_the_clock(self):
        layer = self.make_built_layer(Dropout, input_shape=(100,), rate=0.5)
        inputs = self.xp.ones(100, dtype=self.xp.float32)

        # The layer reads the clock but never steps it; the training loop owns advancement.
        self.assertEqual(layer._clock.value, 0)
        layer.forward(inputs, training=True)
        self.assertEqual(layer._clock.value, 0)
        layer.forward(inputs, training=True)
        self.assertEqual(layer._clock.value, 0)

    def test_active_offset_tracks_the_clock(self):
        layer = self.make_built_layer(Dropout, input_shape=(100,), rate=0.5)
        inputs = self.xp.ones(100, dtype=self.xp.float32)

        layer.forward(inputs, training=True)
        self.assertEqual(layer._active_offset, 0)

        layer._clock.advance()
        layer.forward(inputs, training=True)
        self.assertEqual(layer._active_offset, 1)

    def test_eval_mode_clears_active_offset(self):
        layer = self.make_built_layer(Dropout, input_shape=(100,), rate=0.5)
        inputs = self.xp.ones(100, dtype=self.xp.float32)

        # Run a training pass first so the reset is actually observable.
        layer.forward(inputs, training=True)
        layer.forward(inputs, training=False)

        self.assertEqual(layer._clock.value, 0)
        self.assertEqual(layer._active_offset, -1)

    # ---- regression test for _bind_rng fix -----------------------------------

    def test_explicit_seed_with_stream_id_produces_independent_masks(self):
        """Regression: layers with explicit seed must not share a mask stream.

        Before the fix, two Dropout layers with the same explicit seed (e.g. seed=7)
        at different positions in a model would produce identical masks because
        stream_id was ignored. After the fix, stream_id is always mixed in, so
        masks are independent.

        Note: This test always uses NumPy backend for simplicity (GPU variant skipped).
        """
        # Skip GPU variant; test only on NumPy
        if config.xp != np:
            self.skipTest("Regression test runs only on NumPy backend")

        model = Model()
        model.add(Dense(8, 16))
        model.add(Dropout(rate=0.5, seed=7))
        model.add(Dense(16, 16))
        model.add(Dropout(rate=0.5, seed=7))
        model.manual_seed(999)
        model.finalize((8,))
        model.to('numpy')

        dropout1 = model.layers[1]
        dropout2 = model.layers[3]

        # Regression: explicit seed should take precedence, but stream_id must be mixed in.
        # So two layers at different positions should have different seed keys.
        self.assertNotEqual(
            dropout1._seed_key, dropout2._seed_key,
            msg="Two dropout layers with explicit seed=7 at different positions must have different _seed_key"
        )

        # After a forward pass, their masks should be different.
        inputs = np.random.randn(4, 8).astype(np.float32)
        x = inputs
        for layer in model.layers:
            x = layer.forward(x, training=True)

        self.assertFalse(
            np.all(dropout1.binary_mask == dropout2.binary_mask),
            msg="Two dropout layers must produce different masks"
        )

    def test_explicit_seed_wins_over_model_seed(self):
        """Regression: explicit per-layer seed should take precedence over model seed.

        Note: This test always uses NumPy backend for simplicity (GPU variant skipped).
        """
        # Skip GPU variant; test only on NumPy
        if config.xp != np:
            self.skipTest("Regression test runs only on NumPy backend")

        # Build model with model_seed=100, but explicit layer seed=7 at position 1.
        model1 = Model()
        model1.add(Dense(8, 16))
        model1.add(Dropout(rate=0.5, seed=7))
        model1.add(Dense(16, 16))
        model1.manual_seed(100)
        model1.finalize((8,))
        model1.to('numpy')
        key1 = model1.layers[1]._seed_key

        # Build identical model but with model_seed=200.
        # The Dropout(seed=7) should still have the same _seed_key because
        # explicit seed takes precedence.
        model2 = Model()
        model2.add(Dense(8, 16))
        model2.add(Dropout(rate=0.5, seed=7))
        model2.add(Dense(16, 16))
        model2.manual_seed(200)
        model2.finalize((8,))
        model2.to('numpy')
        key2 = model2.layers[1]._seed_key

        self.assertEqual(
            key1, key2,
            msg="Same explicit seed=7 at same position must produce same _seed_key regardless of model seed"
        )

    # ---- GPU kernel against the reference ------------------------------------

    def _pinned_layer(self, n, rate):
        """make_built_layer's seed only reaches build(), so pin the mask stream directly."""
        layer = self.make_built_layer(Dropout, input_shape=(n,), rate=rate)
        layer._seed_key = self.FIXED_SEED
        return layer

    def test_philox_reference_matches_random123_vectors(self):
        """Published Random123 known answers, so the reference below can be trusted."""
        self.assertEqual(
            philox4x32_10((0, 0, 0, 0), (0, 0)),
            (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8),
        )
        self.assertEqual(
            philox4x32_10((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0)),
            (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1),
        )

    def test_gpu_kernel_writes_nothing_past_the_end(self):
        if not self.uses_gpu_kernel:
            self.skipTest("Philox kernel only runs on the GPU path")

        variant, threads = config.resolve_gpu_launch_geometry()
        kernel = gpu_dropout._get_compiled_kernel(variant, np.dtype(np.float32))
        size = 13
        src = self.xp.ones(size, dtype=self.xp.float32)
        dst = self.xp.full(size + 16, -7.0, dtype=self.xp.float32)

        kernel(
            (1,), (threads,),
            (src, dst, np.uint64(5), np.uint64(3), np.uint32(32768), np.float32(2.0), np.int64(size)),
        )

        self.assertEqual(dst[size:].get().tolist(), [-7.0] * 16)

    def test_keep_threshold_and_scale_at_the_rate_limits(self):
        layer = self.make_built_layer(Dropout, input_shape=(16,), rate=0.1)
        self.assertEqual(layer._keep_threshold, 58982)
        self.assertEqual(layer._gpu_inv_keep, 65536 / 58982)

        # rate 0: 65536 only keeps every draw if the kernel compares it as 32 bits.
        layer = self._pinned_layer(1000, 0.0)
        self.assertEqual(layer._keep_threshold, 65536)
        self.assertEqual(layer._gpu_inv_keep, 1.0)
        inputs = self.xp.linspace(0.1, 5.0, 1000, dtype=self.xp.float32)
        self.assertTrue(bool(self.xp.all(layer.forward(inputs, training=True) == inputs)))

        # Rounding down to 0 would divide by zero.
        layer = self._pinned_layer(1000, 0.999999)
        self.assertEqual(layer._keep_threshold, 1)
        self.assertEqual(layer._gpu_inv_keep, 65536.0)
        output = layer.forward(self.xp.ones(1000, dtype=self.xp.float32), training=True)
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(output))))

    # ---- backward and dtypes -------------------------------------------------

    def test_backward_applies_the_forward_mask_after_the_clock_advances(self):
        n = 40
        layer = self._pinned_layer(n, 0.5)
        kept = layer.forward(self.xp.ones(n, dtype=self.xp.float32), training=True) != 0
        layer._clock.advance()

        dvalues = self.xp.arange(1, n + 1, dtype=self.xp.float32)
        dinputs = layer.backward(dvalues)

        # Hand-derived: rate 0.5 scales kept elements by exactly 2.
        self.assertTrue(bool(self.xp.all(dinputs[kept] == dvalues[kept] * 2.0)))
        self.assertTrue(bool(self.xp.all(dinputs[~kept] == 0.0)))
        self.assertTrue(bool(self.xp.any(kept)) and not bool(self.xp.all(kept)))

    def test_output_follows_input_dtype_with_the_same_mask(self):
        n = 200
        xp = self.xp
        masks = {}
        for dtype in (xp.float16, xp.float32, xp.float64):
            layer = self._pinned_layer(n, 0.5)
            output = layer.forward(xp.full(n, 1.5, dtype=dtype), training=True)
            self.assertEqual(output.dtype, dtype)
            masks[dtype] = output != 0
            self.assertTrue(bool(xp.all(output[masks[dtype]] == 3.0)))

        self.assertTrue(bool(xp.all(masks[xp.float16] == masks[xp.float32])))
        self.assertTrue(bool(xp.all(masks[xp.float64] == masks[xp.float32])))

        # 1 + 2**-40 rounds to 1 in fp32, so a detour through it would lose this.
        layer = self._pinned_layer(n, 0.5)
        output = layer.forward(xp.full(n, 1.0 + 2.0**-40, dtype=xp.float64), training=True)
        self.assertTrue(bool(xp.all(output[masks[xp.float64]] == 2.0 + 2.0**-39)))

    def test_backward_dtype_differing_from_forward_drops_the_same_elements(self):
        n = 200
        layer = self._pinned_layer(n, 0.5)
        kept = layer.forward(self.xp.ones(n, dtype=self.xp.float32), training=True) != 0
        dinputs = layer.backward(self.xp.ones(n, dtype=self.xp.float16))
        self.assertTrue(bool(self.xp.all((dinputs != 0) == kept)))

    def test_non_contiguous_input_matches_its_contiguous_copy(self):
        layer = self._pinned_layer(4, 0.5)
        inputs = self.xp.arange(1, 241, dtype=self.xp.float32).reshape(60, 4)[::2]

        strided = layer.forward(inputs, training=True)
        contiguous = layer.forward(inputs.copy(), training=True)

        self.assertEqual(strided.shape, (30, 4))
        self.assertTrue(bool(self.xp.all(strided == contiguous)))
        kept = strided != 0
        self.assertTrue(bool(self.xp.all(strided[kept] == inputs[kept] * 2.0)))

    def test_dropped_elements_are_zero_even_for_nan_and_inf(self):
        n = 200
        xp = self.xp
        layer = self._pinned_layer(n, 0.5)
        kept = layer.forward(xp.ones(n, dtype=xp.float32), training=True) != 0
        self.assertTrue(bool(xp.any(kept)) and not bool(xp.all(kept)))

        # Every other element is NaN, the rest +inf, so both land in kept and dropped.
        bad = xp.full(n, float('inf'), dtype=xp.float32)
        bad[::2] = float('nan')
        for result in (layer.forward(bad, training=True), layer.backward(bad)):
            self.assertTrue(bool(xp.all(result[~kept] == 0.0)))
            self.assertTrue(bool(xp.all(xp.isnan(result[kept]) == xp.isnan(bad[kept]))))
            self.assertTrue(bool(xp.all(xp.isinf(result[kept]) == xp.isinf(bad[kept]))))

    # ---- precision: the kernel compiles per dtype, so each one needs its own proof ----

    def _expected_gpu_result(self, values, layer, kept):
        """Independent of the kernel: scale in fp32, round to the data dtype, zero the dropped."""
        expected = (values.astype(np.float32) * np.float32(layer._gpu_inv_keep)).astype(values.dtype)
        expected[~kept] = 0
        return expected

    def test_gpu_kernel_matches_reference_in_fp16_and_fp32(self):
        if not self.uses_gpu_kernel:
            self.skipTest("Philox kernel only runs on the GPU path")

        # A wrong element stride or a missing half conversion would change these bytes.
        # 8195 crosses a block boundary at both 512 (HIP) and 1024 (CUDA) threads and ends
        # mid-Philox-call. The large offset exercises the counter's high word. Alternating
        # dtypes checks the cached variants stay separate.
        size = 8195
        offset = 2**32 + 5
        layer = self._pinned_layer(size, 0.1)
        layer._clock.value = offset
        values = np.linspace(-50.0, 50.0, size)
        kept = np.array(kept_elements(layer._seed_key, offset, size, layer._keep_threshold))

        for dtype in (np.float16, np.float32, np.float16, np.float32):
            typed = values.astype(dtype)
            output = layer.forward(self.xp.asarray(typed), training=True)
            self.assertEqual(output.dtype, dtype)
            self.assertEqual(output.get().tobytes(), self._expected_gpu_result(typed, layer, kept).tobytes())

    def test_gpu_model_step_runs_the_kernel_in_the_policy_dtype(self):
        if not self.uses_gpu_kernel:
            self.skipTest("Philox kernel only runs on the GPU path")

        X = self.xp.asarray(np.random.default_rng(1).normal(size=(64, 32)).astype(np.float32))
        y = self.xp.zeros(64, dtype=self.xp.int64)

        for precision, dtype in (('float16', np.float16), ('float32', np.float32)):
            model = Model()
            model.add(Dense(32, 64))
            model.add(ae.ReLU())
            model.add(Dropout(rate=0.2))
            model.add(Dense(64, 2))
            model.configure(
                loss=ae.SoftmaxCategoricalCrossEntropy(),
                optimizer=ae.Adam(lr=0.001),
                accuracy=ae.CategoricalAccuracy(),
            )
            model.to('cupy')
            model.manual_seed(seed=42)
            model.set_precision(compute_dtype=precision)
            model.finalize(input_shape=(32,))
            relu, dropout, next_dense = model.layers[1:4]

            out = model.forward(X, training=True)
            model.loss.calculate(out, y)
            model.loss.backward(out, y)
            model.backward(model.loss.dinputs)

            self.assertEqual(dropout.forward.__name__, '_forward_gpu')
            inputs, dvalues = relu.output.get(), next_dense.dinputs.get()
            self.assertEqual(inputs.dtype, dtype)
            self.assertEqual(dvalues.dtype, dtype)

            kept = np.array(
                kept_elements(dropout._seed_key, dropout._active_offset, inputs.size, dropout._keep_threshold)
            ).reshape(inputs.shape)
            self.assertEqual(
                dropout.output.get().tobytes(), self._expected_gpu_result(inputs, dropout, kept).tobytes()
            )
            self.assertEqual(
                dropout.dinputs.get().tobytes(), self._expected_gpu_result(dvalues, dropout, kept).tobytes()
            )
            self.assertTrue(bool((dropout.dinputs.get()[kept] != 0).any()))

base_case.register_test_suites(globals(), TestDropout)