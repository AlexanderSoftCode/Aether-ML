import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.custom_kernels import conv_kernel


class TestConvBackwardCastKernel(base_case.AetherBaseTestCase):
    """conv_backward_cast_fp16: dvalues scaled by 2^10 then clipped to the fp16 range, inputs neither."""

    def setUp(self):
        super().setUp()
        if self.backend_name != "cupy":
            self.skipTest("CuPy backend required")
        if conv_kernel.get_compiled_backward_cast_kernel(config.IS_HIP) is None:
            self.skipTest("conv backward cast kernel unavailable on this device")

    def _cast(self, dvalues, inputs):
        result = conv_kernel.cast_backward_operands_fp16(
            self.xp.asarray(dvalues, dtype=self.xp.float32),
            self.xp.asarray(inputs, dtype=self.xp.float32),
        )
        self.assertIsNotNone(result)
        return [self.xp.asnumpy(r) for r in result]

    def test_special_values(self):
        inf, nan = np.inf, np.nan
        dvalues = [-inf, -70.0, -64.0, -1.5, 1e-8, 0.0, 63.96875, 64.0, 70000.0, inf, nan]
        inputs = [70000.0, -70000.0, 0.1, nan, 2.0]
        dv16, in16 = self._cast(dvalues, inputs)

        self.assertEqual(dv16.dtype, np.float16)
        self.assertEqual(in16.dtype, np.float16)
        # 1e-8 * 2^10 is 171.8 subnormal steps of 2^-24, which rounds to 172.
        np.testing.assert_array_equal(
            dv16,
            np.array([-65504, -65504, -65504, -1536, 172 * 2.0 ** -24, 0, 65504, 65504, 65504, 65504, nan],
                     dtype=np.float16),
        )
        # Unscaled and unclipped: out-of-range inputs overflow to inf exactly as forward's astype does.
        np.testing.assert_array_equal(
            in16, np.array([inf, -inf, np.float16(0.1), nan, 2.0], dtype=np.float16)
        )

    def test_matches_scale_clip_and_astype(self):
        """Both ranges, with lengths that end mid-block, against NumPy's scale + clip + astype."""
        rng = np.random.default_rng(0)
        dvalues = (rng.standard_normal((3, 7, 5, 19)) * 30).astype(np.float32)
        inputs = (rng.standard_normal((3, 9, 9, 7)) * 3e4).astype(np.float32)
        dv16, in16 = self._cast(dvalues, inputs)

        self.assertEqual(dv16.shape, dvalues.shape)
        self.assertEqual(in16.shape, inputs.shape)
        self.assertTrue((np.abs(dvalues) * 2 ** 10 > 65504).any(), "test data must exercise the clip")
        self.assertTrue((np.abs(inputs) > 65504).any(), "test data must exercise the overflow")
        np.testing.assert_array_equal(dv16, np.clip(dvalues * 2 ** 10, -65504, 65504).astype(np.float16))
        # The overflow to inf is the behavior under test, so NumPy's warning is expected.
        with np.errstate(over="ignore"):
            expected_in16 = inputs.astype(np.float16)
        np.testing.assert_array_equal(in16, expected_in16)


base_case.register_test_suites(globals(), template_cls=TestConvBackwardCastKernel)
