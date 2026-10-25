import warnings

import aether.config as config
import tests.base_case as base_case
from aether.layers.conv import Conv2d
from tests.layers.conv2d.base import ConvLayerTestBase


class TestConvPrecision(ConvLayerTestBase):
    """Precision-policy handling for a layer whose kernels pin their own dtype."""

    @staticmethod
    def _capture(fn):
        """Runs fn and returns every UserWarning it emitted."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
        return [str(w.message) for w in caught]

    def _make_uncompiled_conv(self):
        """A Conv2d that has never seen _compile_for_device."""
        return Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
        )

    def _apply(self, layer, compute_dtype):
        return lambda: layer._apply_precision(config.DTypePolicy(compute_dtype))

    def test_policy_matching_the_bound_path_is_silent(self):
        """Asking for exactly what the bound kernels already do isn't a mismatch."""
        matching = self.layer._effective_compute_dtype()
        self.assertEqual(self._capture(self._apply(self.layer, matching)), [])

    def test_mismatched_policy_warns_once_naming_both_dtypes(self):
        """float64 is never an effective dtype, so it mismatches on either path."""
        effective = self.layer._effective_compute_dtype()
        caught = self._capture(self._apply(self.layer, "float64"))

        self.assertEqual(len(caught), 1)
        self.assertIn("float64", caught[0])
        self.assertIn(effective, caught[0])
        self.assertIn("Conv2d", caught[0])

    def test_repeated_compile_does_not_repeat_the_warning(self):
        """Model.to() then Model.finalize() both call _compile_for_device."""
        layer = self._make_uncompiled_conv()
        layer._apply_precision(config.DTypePolicy("float64"))

        caught = self._capture(
            lambda: [layer._compile_for_device(self.backend_name) for _ in range(2)]
        )
        self.assertEqual(len(caught), 1)

    def test_precision_set_before_compile_defers_to_compile(self):
        """set_precision() can precede to(); there's no binding to check yet."""
        layer = self._make_uncompiled_conv()

        before = self._capture(self._apply(layer, "float64"))
        self.assertEqual(before, [])

        after = self._capture(lambda: layer._compile_for_device(self.backend_name))
        self.assertEqual(len(after), 1)

    def test_ignored_policy_is_still_stored_for_serialization(self):
        """The warning must not stop the policy reaching Model.save()'s manifest."""
        self._capture(self._apply(self.layer, "float64"))

        self.assertEqual(self.layer.precision_policy.compute_dtype_name, "float64")

    def test_layer_still_computes_under_an_ignored_policy(self):
        """Warn-and-continue: output is unchanged, not an exception."""
        self._capture(self._apply(self.layer, "float64"))

        out = self.layer.forward(self.test_images, training=True)
        self.assertEqual(out.shape, (self.CONV_SHAPE[0], 28, 28, self.OUT_CHANNELS))
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(out))))


base_case.register_test_suites(globals(), TestConvPrecision)
