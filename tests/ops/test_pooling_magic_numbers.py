import numpy as np

import tests.base_case as base_case
from aether.custom_kernels.launch_math import _compute_magic_numbers


class TestPoolingMagicNumbers(base_case.AetherBaseTestCase):
    """
    Regression suite verifying O(1) fast division magic numbers
    against the Hacker's Delight Ch. 10 rounding trap.
    """

    def to_cpu(self, array):
        """Safely convert any backend array (NumPy or CuPy) to a standard CPU NumPy ndarray."""
        if hasattr(array, 'get'):  # CuPy ndarray
            return array.get()
        if hasattr(self, 'xp') and hasattr(self.xp, 'asnumpy'):  # CuPy module fallback
            return self.xp.asnumpy(array)
        return np.asarray(array)

    def test_magic_number_ch10_trap(self):
        h_out_values = [1, 2, 3, 5, 7, 11, 13, 14, 28, 31, 56, 112, 224]
        s_values = [1, 4, 8, 32, 64]
        block_z_values = [1, 4, 8, 16]

        for H_out in h_out_values:
            for S in s_values:
                for block_z in block_z_values:
                    with self.subTest(H_out=H_out, S=S, block_z=block_z):
                        grid_z = (H_out * S + block_z - 1) // block_z
                        max_h_s = grid_z * block_z - 1

                        scale, shift = _compute_magic_numbers(H_out, max_h_s)

                        u_scale = self.xp.uint64(scale)
                        u_d = self.xp.uint64(H_out)
                        shift_amount = 32 + int(shift)

                        rem = (max_h_s + 1) % H_out
                        n_bad = max_h_s - rem

                        critical_indices = [0, H_out - 1, H_out, max_h_s]
                        if n_bad >= 0:
                            critical_indices.append(n_bad)

                        # De-duplicate indices safely on CPU
                        unique_indices = np.unique(critical_indices)
                        test_vec = self.xp.array(unique_indices, dtype=self.xp.uint64)

                        approx = (test_vec * u_scale) >> shift_amount
                        expected = test_vec // u_d

                        np.testing.assert_array_equal(
                            self.to_cpu(approx),
                            self.to_cpu(expected),
                            err_msg=(
                                f"[{self.backend_name.upper()}] Off-by-one division error for "
                                f"H_out={H_out}, max_h_s={max_h_s}! scale={scale}, shift={shift}"
                            ),
                        )

    def test_exhaustive_range(self):
        odd_divisors = [3, 7, 9, 13, 15, 27, 31, 55, 63]
        max_h_s = 2048

        for H_out in odd_divisors:
            with self.subTest(H_out=H_out):
                scale, shift = _compute_magic_numbers(H_out, max_h_s)

                n = self.xp.arange(max_h_s + 1, dtype=self.xp.uint64)
                shift_amount = 32 + int(shift)
                approx = (n * self.xp.uint64(scale)) >> shift_amount
                expected = n // self.xp.uint64(H_out)

                np.testing.assert_array_equal(
                    self.to_cpu(approx),
                    self.to_cpu(expected),
                    err_msg=f"[{self.backend_name.upper()}] Exhaustive check failed for H_out={H_out}",
                )

    def test_gpu_shift_bounds(self):
        """Verify shift_amount stays within legal CUDA/HIP bit-shift bounds [0, 63]."""
        for d in [1, 2, 3, 7, 16, 255, 1024, 65535]:
            for max_num in [1, 100, 10000, 2**31 - 1]:
                _, shift = _compute_magic_numbers(d, max_num)
                shift_amount = 32 + int(shift)

                self.assertGreaterEqual(shift_amount, 0, f"UB: Negative shift for d={d}")
                self.assertLess(shift_amount, 64, f"UB: Shift >= 64 for d={d}")

    def test_boundary_numerators(self):
        """Test zero, sub-divisor ranges, and 32-bit ceiling values."""
        cases = [
            (7, 0),
            (64, 31),
            (3, 2**31 - 1),
        ]
        for d, max_num in cases:
            with self.subTest(d=d, max_num=max_num):
                scale, shift = _compute_magic_numbers(d, max_num)
                shift_amount = 32 + int(shift)

                for n in [0, max_num // 2, max_num]:
                    approx = (n * int(scale)) >> shift_amount
                    self.assertEqual(approx, n // d)

    def test_power_of_two_divisors(self):
        """Verify power-of-two pooling strides yield exact divisions."""
        for p in range(1, 16):
            d = 1 << p
            max_num = 10000
            scale, shift = _compute_magic_numbers(d, max_num)

            n = np.arange(max_num + 1, dtype=np.uint64)
            shift_amount = 32 + int(shift)
            approx = (n * np.uint64(scale)) >> shift_amount
            expected = n // d

            np.testing.assert_array_equal(approx, expected)

    def test_randomized_fuzzing(self):
        """Fuzz random (d, max_num) pairs to catch unexpected magic number failures."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            d = int(rng.integers(1, 1024))
            max_num = int(rng.integers(1, 100000))

            scale, shift = _compute_magic_numbers(d, max_num)
            shift_amount = 32 + int(shift)

            test_points = np.unique(np.concatenate([
                [0, max_num],
                rng.integers(0, max_num + 1, size=100)
            ])).astype(np.uint64)

            approx = (test_points * np.uint64(scale)) >> shift_amount
            expected = test_points // np.uint64(d)

            np.testing.assert_array_equal(approx, expected, err_msg=f"Failed for d={d}, max={max_num}")

    def test_d1_negative_shift_kernel_execution(self):
        """
        Regression test: Verifies H_out=1 (d=1) returns shift=-32 and compiles/executes
        correctly in C++ without unsigned wrap-around errors.
        """
        H_out = 1
        max_h_s = 63
        scale, shift = _compute_magic_numbers(H_out, max_h_s)

        self.assertEqual(int(shift), -32, f"Expected shift=-32 for d=1, got {shift}")
        self.assertEqual(32 + int(shift), 0, f"Expected (32 + shift) == 0, got {32 + int(shift)}")

        if self.backend_name == 'cupy':
            kernel_code = r'''
            extern "C" __global__
            void test_shift_kernel(
                const unsigned int magic_scale,
                const int magic_shift,
                int* out
            ) {
                int h_s = threadIdx.x;
                unsigned long long prod = (unsigned long long)h_s * magic_scale;
                out[h_s] = (int)(prod >> (32 + magic_shift));
            }
            '''
            test_kernel = self.xp.RawKernel(kernel_code, 'test_shift_kernel')

            n_threads = 16
            out = self.xp.zeros(n_threads, dtype=self.xp.int32)
            test_kernel((1,), (n_threads,), (self.xp.uint32(scale), self.xp.int32(shift), out))

            expected = np.arange(n_threads, dtype=np.int32)
            np.testing.assert_array_equal(
                self.to_cpu(out),
                expected,
                err_msg="C++ kernel produced incorrect output for shift=-32"
            )


base_case.register_test_suites(globals(), TestPoolingMagicNumbers)
