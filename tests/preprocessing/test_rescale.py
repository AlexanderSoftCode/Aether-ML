import numpy as np

import tests.base_case as base_case
from aether.preprocessing.transforms import Rescale


class TestRescaleTransform(base_case.AetherBaseTestCase):

    def test_uint8_single_arg_does_not_promote_to_float64(self):
        X = self.xp.arange(0, 255, 5, dtype=self.xp.uint8)
        out = self.make_component(Rescale, factor=1.0 / 255.0).transform(X)
        self.assertEqual(out.dtype, np.float32)

    def test_uint8_scaled_values_correct(self):
        X = self.xp.array([0, 128, 255], dtype=self.xp.uint8)
        out = self.make_component(Rescale, factor=1.0 / 255.0).transform(X)
        expected = X.astype(self.xp.float32) * np.float32(1.0 / 255.0)
        self.assertTrue(self.xp.allclose(out, expected))

    def test_signed_integer_input_promotes_to_float32(self):
        X = self.xp.array([-10, 0, 10], dtype=self.xp.int16)
        out = self.make_component(Rescale, factor=0.5).transform(X)
        self.assertEqual(out.dtype, np.float32)

    def test_float_input_dtype_untouched(self):
        # Already-float input keeps its own precision -- the guard only
        # fires for integer-kind input.
        X = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float64)
        out = self.make_component(Rescale, factor=2.0).transform(X)
        self.assertEqual(out.dtype, np.float64)

    def test_call_alias_matches_transform(self):
        X = self.xp.array([0, 255], dtype=self.xp.uint8)
        rescale = self.make_component(Rescale, factor=1.0 / 255.0)
        self.assertTrue(self.xp.allclose(rescale(X), rescale.transform(X)))

    def test_multi_arg_skips_integer_labels(self):
        X = self.xp.array([0.0, 255.0], dtype=self.xp.float32)
        y = self.xp.array([0, 1], dtype=self.xp.int64)
        out_X, out_y = self.make_component(Rescale, factor=1.0 / 255.0).transform(X, y)
        self.assertTrue(self.xp.allclose(out_X, X * np.float32(1.0 / 255.0)))
        self.assertTrue(self.xp.array_equal(out_y, y))

    def test_multi_arg_uint8_feature_left_unscaled(self):
        # Pre-existing (unchanged) behavior: the multi-arg path only scales
        # float-kind arrays by design, so a raw uint8 "X" passed alongside a
        # label array is left untouched rather than scaled. Pinned here so
        # it's a visible, intentional-looking contract rather than a silent gap.
        X = self.xp.array([0, 255], dtype=self.xp.uint8)
        y = self.xp.array([0, 1], dtype=self.xp.int64)
        out_X, _ = self.make_component(Rescale, factor=1.0 / 255.0).transform(X, y)
        self.assertTrue(self.xp.array_equal(out_X, X))

    def test_empty_args_returns_none(self):
        self.assertIsNone(self.make_component(Rescale).transform())

    def test_get_config(self):
        self.assertEqual(self.make_component(Rescale, factor=0.5).get_config(), {"factor": 0.5})

    def test_is_fitted_always_true(self):
        self.assertTrue(self.make_component(Rescale).is_fitted)


base_case.register_test_suites(globals(), TestRescaleTransform)