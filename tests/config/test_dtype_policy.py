import warnings

import numpy as np

import tests.base_case as base_case
from aether.config import PARAM_DTYPE, DTypePolicy


class TestDTypePolicy(base_case.AetherBaseTestCase):

    def setUp(self):
        super().setUp()

        # Suppress NumPy float16 emulation warnings only during DTypePolicy testing
        self._warn_ctx = warnings.catch_warnings()
        self._warn_ctx.__enter__()
        warnings.filterwarnings(
            "ignore",
            message=r".*NumPy float16 is emulated.*",
            category=UserWarning
        )

    def tearDown(self):
        # Clean up the warning context first
        if hasattr(self, "_warn_ctx"):
            self._warn_ctx.__exit__(None, None, None)

        super().tearDown()

    # ---- Initialization Guardrails --------------

    def test_invalid_string_raises_value_error(self):
        """Ensure passing strings outside COMPUTE_DTYPE raises ValueError."""
        invalid_dtypes = ['int8', 'int32', 'float8', 'complex64', 'invalid_type', 'bfloat16']
        for dt in invalid_dtypes:
            with self.subTest(dtype=dt), self.assertRaises(ValueError):
                DTypePolicy(compute_dtype=dt)

    def test_non_string_non_none_raises_type_error(self):
        """Ensure non-str and non-None inputs raise TypeError."""
        invalid_types = [123, 32.0, ['float32'], {'dtype': 'float32'}, True]
        for val in invalid_types:
            with self.subTest(val=val), self.assertRaises(TypeError):
                DTypePolicy(compute_dtype=val)

    def test_valid_initializations(self):
        """Verify valid compute_dtypes set attributes correctly."""
        valid_dtypes = [None, 'float16', 'float32', 'float64']
        for dt in valid_dtypes:
            with self.subTest(dtype=dt):
                policy = DTypePolicy(compute_dtype=dt)
                self.assertEqual(policy.compute_dtype_name, dt)
                self.assertEqual(policy.param_dtype, np.dtype(PARAM_DTYPE))

    # ---- Casting Integrity --------------

    def test_cast_to_compute_none_returns_identity(self):
        """Verify cast_to_compute returns unchanged reference when compute_dtype is None."""
        policy = DTypePolicy(compute_dtype=None)
        x = self.xp.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = self.xp.array([4.0, 5.0, 6.0], dtype=np.float64)

        out_x, out_y = policy.cast_to_compute(x, y)
        self.assertIs(out_x, x)
        self.assertIs(out_y, y)

    def test_cast_to_compute_converts_precision(self):
        """Verify cast_to_compute casts arrays to the configured compute precision."""
        policy = DTypePolicy(compute_dtype='float16')
        x = self.xp.array([1.0, 2.0, 3.0], dtype=np.float32)

        out_x = policy.cast_to_compute(x)
        self.assertEqual(out_x.dtype, self.xp.float16)
        self.assertEqual(out_x.shape, x.shape)

    def test_cast_to_param_restores_float32(self):
        """Verify cast_to_param strictly converts float16/float64 tensors back to float32."""
        policy = DTypePolicy(compute_dtype='float16')
        grad_fp16 = self.xp.array([0.1, 0.2, 0.3], dtype=np.float16)
        grad_fp64 = self.xp.array([0.4, 0.5, 0.6], dtype=np.float64)

        out_16, out_64 = policy.cast_to_param(grad_fp16, grad_fp64)
        self.assertEqual(out_16.dtype, self.xp.float32)
        self.assertEqual(out_64.dtype, self.xp.float32)

    def test_none_tensor_arguments_handled_safely(self):
        """Verify passing None (e.g. optional bias gradients) preserves None."""
        policy = DTypePolicy(compute_dtype='float16')
        weight_tensor = self.xp.array([1.0, 2.0], dtype=np.float32)
        bias_tensor = None

        res_w, res_b = policy.cast_to_compute(weight_tensor, bias_tensor)
        self.assertEqual(res_w.dtype, self.xp.float16)
        self.assertIsNone(res_b)

        res_pw, res_pb = policy.cast_to_param(res_w, bias_tensor)
        self.assertEqual(res_pw.dtype, self.xp.float32)
        self.assertIsNone(res_pb)

    # ==========================================
    # Casting & Unpacking Mechanics Tests
    # ==========================================

    def test_cast_empty_tensors_returns_empty_tuple(self):
        """Verify calling casting methods with no arguments returns an empty tuple."""
        policy = DTypePolicy(compute_dtype='float16')
        self.assertEqual(policy.cast_to_compute(), ())
        self.assertEqual(policy.cast_to_param(), ())

    def test_cast_to_compute_single_tensor_unpacking(self):
        """Verify single argument returns bare tensor without requiring tuple unpack."""
        policy = DTypePolicy(compute_dtype='float16')
        x = self.xp.array([1.0, 2.0, 3.0], dtype=np.float32)

        out_x = policy.cast_to_compute(x)
        self.assertNotIsInstance(out_x, tuple)
        self.assertEqual(out_x.dtype, self.xp.float16)
        self.assertEqual(out_x.shape, x.shape)

    def test_cast_to_compute_multi_tensor_unpacking(self):
        """Verify multi-argument returns a tuple of converted tensors."""
        policy = DTypePolicy(compute_dtype='float16')
        x = self.xp.array([1.0, 2.0, 3.0], dtype=np.float32)
        w = self.xp.array([3.0, 4.0], dtype=np.float32)

        res = policy.cast_to_compute(x, w)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].dtype, self.xp.float16)
        self.assertEqual(res[1].dtype, self.xp.float16)

    def test_cast_to_param_single_and_multi_unpacking(self):
        """Verify cast_to_param unwraps single tensors and returns tuples for multiple."""
        policy = DTypePolicy(compute_dtype='float16')
        grad_fp16 = self.xp.array([0.1, 0.2, 0.3], dtype=np.float16)
        grad_fp64 = self.xp.array([0.4, 0.5, 0.6], dtype=np.float64)

        # Single arg test (should not return a tuple)
        single_out = policy.cast_to_param(grad_fp16)
        self.assertNotIsInstance(single_out, tuple)
        self.assertEqual(single_out.dtype, self.xp.float32)

        # Multi arg test (should return a tuple)
        out_16, out_64 = policy.cast_to_param(grad_fp16, grad_fp64)
        self.assertEqual(out_16.dtype, self.xp.float32)
        self.assertEqual(out_64.dtype, self.xp.float32)

    def test_single_none_tensor_handled_safely(self):
        """Verify passing a single None argument returns None directly."""
        policy = DTypePolicy(compute_dtype='float16')
        self.assertIsNone(policy.cast_to_compute(None))
        self.assertIsNone(policy.cast_to_param(None))


base_case.register_test_suites(globals(), TestDTypePolicy)
