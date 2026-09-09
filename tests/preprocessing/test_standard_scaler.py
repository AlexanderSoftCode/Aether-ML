import numpy as np
import warnings
import aether.config as config
import tests.base_case as base_case

from aether.preprocessing.transforms import StandardScaler


class TestStandardScalerTransform(base_case.AetherBaseTestCase):

    def test_unfit_scaler_raises_error(self):
        """Calling __call__ without prior fit() or pre-computed stats must raise ValueError."""
        scaler = StandardScaler()
        X = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float32)

        with self.assertRaises(ValueError):
            scaler(X)

    def test_fit_calculates_correct_statistics_and_chains(self):
        """fit() should compute accurate mean/std and return self for method chaining."""
        scaler = StandardScaler()
        X = self.xp.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=self.xp.float32)

        returned_instance = scaler.fit(X)

        self.assertIs(returned_instance, scaler)

        self.assertAlmostEqual(float(scaler.mean.item()), 30.0, places=4)
        self.assertAlmostEqual(float(scaler.std.item()), 14.1421356, places=4)

    def test_fit_integer_dtype_conversion(self):
        """Integer input arrays should default target dtype to float32 upon fitting."""
        scaler = StandardScaler()
        X_int = self.xp.array([0, 128, 255], dtype=self.xp.uint8)

        scaler.fit(X_int)

        self.assertEqual(scaler.dtype, np.float32)

    def test_standardization_zero_mean_unit_variance(self):
        """Scaling data should yield approximately zero mean and unit variance."""
        scaler = StandardScaler()

        X = self.xp.random.randn(1000, 16).astype(self.xp.float32) * 5.0 + 10.0

        scaler.fit(X)
        X_scaled = scaler(X)

        scaled_mean = float(self.xp.mean(X_scaled))
        scaled_std = float(self.xp.std(X_scaled))

        self.assertAlmostEqual(scaled_mean, 0.0, places=3)
        self.assertAlmostEqual(scaled_std, 1.0, places=3)

    def test_precomputed_mean_and_std_initialization(self):
        """Initializing with explicit mean/std allows scaling without running fit()."""
        scaler = StandardScaler(mean=10.0, std=2.0)
        scaler.dtype = self.xp.float32

        X = self.xp.array([10.0, 12.0, 6.0], dtype=self.xp.float32)
        X_scaled = scaler(X)

        # (10 - 10) / 2 = 0;  (12 - 10) / 2 = 1;  (6 - 10) / 2 = -2
        expected = self.xp.array([0.0, 1.0, -2.0], dtype=self.xp.float32)

        self.assertTrue(self.xp.allclose(X_scaled, expected, atol=1e-5))

    def test_input_dtype_coercion_on_call(self):
        """Inputs with a dtype mismatched from fitted scaler.dtype should be safely casted."""
        scaler = StandardScaler()
        X_train = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float32)
        scaler.fit(X_train)

        # Pass mismatched float64 array during inference
        X_test_float64 = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float64)
        X_scaled = scaler(X_test_float64)

        self.assertEqual(str(X_scaled.dtype), 'float32')

    def test_standard_scaler_default_axis_none(self):
        """Verify that default axis=None computes global scalar mean/std while keeping dimensions."""
        scaler = StandardScaler(axis=None)
        X = self.xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.xp.float32)
        scaler.fit(X)

        # Global mean is (1 + 2 + 3 + 4) / 4 = 2.5
        self.assertEqual(scaler.mean.shape, (1, 1))
        self.assertEqual(scaler.std.shape, (1, 1))
        self.assertAlmostEqual(float(scaler.mean[0, 0]), 2.5, places=5)

        out = scaler(X)
        self.assertEqual(out.shape, X.shape)
        self.assertAlmostEqual(float(self.xp.mean(out)), 0.0, places=5)
        self.assertAlmostEqual(float(self.xp.std(out)), 1.0, places=5)

    def test_standard_scaler_tabular_axis_zero(self):
        """Verify feature-wise standardization for 2D tabular data using axis=0."""
        scaler = StandardScaler(axis=0)
        # 2 samples, 3 features
        X = self.xp.array([
            [10.0, 100.0, 1.0],
            [20.0, 200.0, 3.0]
        ], dtype=self.xp.float32)
        scaler.fit(X)

        # Feature means: [15.0, 150.0, 2.0]
        self.assertEqual(scaler.mean.shape, (1, 3))
        self.assertEqual(scaler.std.shape, (1, 3))
        expected_means = self.xp.array([[15.0, 150.0, 2.0]], dtype=self.xp.float32)
        self.assertTrue(self.xp.allclose(scaler.mean, expected_means, atol=1e-5))

        out = scaler(X)
        # Each column should now have zero mean
        col_means = self.xp.mean(out, axis=0)
        self.assertTrue(self.xp.allclose(col_means, self.xp.zeros((3,), dtype=self.xp.float32), atol=1e-5))

    def test_standard_scaler_channel_wise_image_axis(self):
        """Verify channel-wise standardization for 4D image batches using axis=(0, 1, 2)."""
        scaler = StandardScaler(axis=(0, 1, 2))
        # Batch of 4 images, 8x8, 3 channels (RGB)
        X = self.xp.random.uniform(0.0, 1.0, size=(4, 8, 8, 3)).astype(self.xp.float32)
        scaler.fit(X)

        # Shape should preserve rank with shape (1, 1, 1, 3)
        self.assertEqual(scaler.mean.shape, (1, 1, 1, 3))
        self.assertEqual(scaler.std.shape, (1, 1, 1, 3))

        out = scaler(X)
        self.assertEqual(out.shape, (4, 8, 8, 3))
        self.assertEqual(str(out.dtype), 'float32')

        # Channel means across (batch, height, width) should be zero
        channel_means = self.xp.mean(out, axis=(0, 1, 2))
        self.assertTrue(self.xp.allclose(channel_means, self.xp.zeros((3,), dtype=self.xp.float32), atol=1e-5))
        
    def test_is_fitted_false_before_fit(self):
        self.assertFalse(self.make_component(StandardScaler).is_fitted)
 
    def test_is_fitted_true_after_fit(self):
        X = self.xp.arange(24, dtype=self.xp.float32).reshape(4, 6)
        scaler = self.make_component(StandardScaler).fit(X)
        self.assertTrue(scaler.is_fitted)
 
    def test_transform_before_fit_raises(self):
        X = self.xp.ones((2, 2), dtype=self.xp.float32)
        with self.assertRaises(ValueError):
            self.make_component(StandardScaler).transform(X)
 
    def test_fit_computes_correct_statistics(self):
        X = self.xp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=self.xp.float32)
        scaler = self.make_component(StandardScaler, axis=0).fit(X)
        self.assertTrue(self.xp.allclose(scaler.mean, self.xp.mean(X, axis=0, keepdims=True)))
        self.assertTrue(self.xp.allclose(scaler.std, self.xp.std(X, axis=0, keepdims=True)))
 
    def test_transform_matches_manual_standardization(self):
        X = self.xp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=self.xp.float32)
        scaler = self.make_component(StandardScaler, axis=0).fit(X)
        out = scaler.transform(X)
        expected = (X - scaler.mean) / (scaler.std + 1e-8)
        self.assertTrue(self.xp.allclose(out, expected))
 
    def test_transform_casts_integer_input_to_float32(self):
        X = (self.xp.arange(24) % 10).astype(self.xp.uint8).reshape(4, 6)
        scaler = self.make_component(StandardScaler, axis=0).fit(X)
        self.assertEqual(scaler.dtype, np.float32)
        self.assertEqual(scaler.transform(X).dtype, np.float32)
 
    def test_constructor_coerces_list_mean_std(self):
        scaler = StandardScaler(mean=[0.0, 0.0], std=[1.0, 1.0])
        self.assertTrue(scaler.is_fitted)
        self.assertTrue(hasattr(scaler.mean, "dtype"))
        self.assertTrue(hasattr(scaler.std, "dtype"))
 
    def test_constructor_leaves_live_array_untouched(self):
        mean = self.xp.array([0.0, 0.0], dtype=self.xp.float32)
        scaler = StandardScaler(mean=mean, std=self.xp.array([1.0, 1.0], dtype=self.xp.float32))
        self.assertIs(scaler.mean, mean)
 
    def test_get_config_roundtrip(self):
        X = self.xp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=self.xp.float32)
        scaler = self.make_component(StandardScaler, axis=0).fit(X)
        cfg = scaler.get_config()
 
        # from_config, not StandardScaler(**cfg): the config carries the dtype-pin
        # flag, which is derived state rather than a constructor argument.
        restored = StandardScaler.from_config(cfg)
        self.assertTrue(restored.is_fitted)
        self.assertEqual(restored.dtype, scaler.dtype)
        self.assertTrue(self.xp.allclose(restored.transform(X), scaler.transform(X)))
 
    def test_get_config_before_fit_is_null(self):
        cfg = self.make_component(StandardScaler).get_config()
        self.assertIsNone(cfg["mean"])
        self.assertIsNone(cfg["std"])
 
    def test_get_config_axis_tuple_becomes_list(self):
        X = self.xp.zeros((2, 3, 3, 4), dtype=self.xp.float32)
        cfg = self.make_component(StandardScaler, axis=(0, 1, 2)).fit(X).get_config()
        self.assertEqual(cfg["axis"], [0, 1, 2])
 
    def test_compile_for_device_noop_when_unfit(self):
        # make_component constructs then immediately calls _compile_for_device --
        # must not crash on mean/std still being None.
        scaler = self.make_component(StandardScaler, axis=0)
        self.assertFalse(scaler.is_fitted)
 
    def test_compile_for_device_migrates_fitted_stats(self):
        X = self.xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.xp.float32)
        scaler = self.make_component(StandardScaler, axis=0).fit(X)
        before = self.xp.asarray(scaler.mean).copy()
 
        scaler._compile_for_device(self.backend_name)
 
        self.assertTrue(self.xp.allclose(scaler.mean, before))
 
    def test_apply_precision_overrides_dtype(self):
        X = self.xp.array([[1, 2, 3], [4, 5, 6]], dtype=self.xp.uint8)
        scaler = self.make_component(StandardScaler, axis=0).fit(X)
        self.assertEqual(scaler.dtype, np.float32)
 
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*NumPy float16 is emulated.*",
                category=UserWarning,
            )
            policy = config.DTypePolicy(compute_dtype="float16")
            scaler._apply_precision(policy)
 
        self.assertEqual(scaler.dtype, np.dtype("float16"))
 
    def test_apply_precision_noop_when_policy_has_no_compute_dtype(self):
        X = self.xp.array([[1, 2, 3], [4, 5, 6]], dtype=self.xp.uint8)
        scaler = self.make_component(StandardScaler, axis=0).fit(X)
        self.assertEqual(scaler.dtype, np.float32)
 
        scaler._apply_precision(config.DTypePolicy(compute_dtype=None))
 
        self.assertEqual(scaler.dtype, np.float32)


    # ---- dtype ladder: constructor pin > precision policy > fit inference ----

    def test_constructor_pin_outranks_precision_policy(self):
        scaler = StandardScaler(dtype="float32")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*NumPy float16 is emulated.*", category=UserWarning
            )
            scaler._apply_precision(config.DTypePolicy(compute_dtype="float16"))

        self.assertEqual(scaler.dtype, np.dtype("float32"))

    def test_constructor_pin_survives_fit(self):
        """fit() may only infer -- it must never overwrite a pin."""
        scaler = StandardScaler(dtype="float32")
        scaler.fit(self.xp.ones((4, 3), dtype=self.xp.float64))

        self.assertEqual(scaler.dtype, np.dtype("float32"))
        self.assertEqual(scaler.mean.dtype, np.dtype("float32"))
        self.assertEqual(scaler.std.dtype, np.dtype("float32"))

    def test_precision_policy_outranks_fit_inference(self):
        # _apply_precision remains the documented hook: the exemption only stops
        # the model from dispatching, a direct call still applies.
        scaler = StandardScaler()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*NumPy float16 is emulated.*", category=UserWarning
            )
            scaler._apply_precision(config.DTypePolicy(compute_dtype="float16"))

        scaler.fit(self.xp.ones((4, 3), dtype=self.xp.float64))
        self.assertEqual(scaler.dtype, np.dtype("float16"))

    def test_fit_inference_applies_when_nothing_is_pinned(self):
        scaler = StandardScaler()
        scaler.fit(self.xp.ones((4, 3), dtype=self.xp.float64))
        self.assertEqual(np.dtype(scaler.dtype), np.dtype("float64"))

    def test_apply_precision_after_fit_recasts_statistics(self):
        scaler = StandardScaler(axis=0).fit(
            self.xp.arange(12, dtype=self.xp.float32).reshape(4, 3)
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*NumPy float16 is emulated.*", category=UserWarning
            )
            scaler._apply_precision(config.DTypePolicy(compute_dtype="float16"))

        self.assertEqual(scaler.mean.dtype, np.dtype("float16"))
        self.assertEqual(scaler.std.dtype, np.dtype("float16"))

    def test_dtype_pin_below_float32_is_rejected(self):
        """Statistics are never computed/stored below single (float32) precision."""
        with self.assertRaises(ValueError):
            StandardScaler(dtype="float16")

    def test_fit_promotes_a_float16_input_to_float32(self):
        """An unpinned scaler fit on float16 data still stores stats at >= float32."""
        scaler = StandardScaler()
        scaler.fit(self.xp.ones((4, 3), dtype=self.xp.float16))

        self.assertEqual(scaler.mean.dtype, np.dtype("float32"))
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(scaler.std))))

    # ---- dtype traps on the round-trip ----

    def test_restored_statistics_are_float32_not_float64(self):
        """get_config emits nested lists; a bare np.asarray would return float64."""
        X = self.xp.arange(12, dtype=self.xp.float32).reshape(4, 3)
        restored = StandardScaler.from_config(
            StandardScaler(axis=0).fit(X).get_config()
        )

        self.assertEqual(restored.mean.dtype, np.dtype("float32"))
        self.assertEqual(restored.std.dtype, np.dtype("float32"))

    def test_list_statistics_without_a_dtype_coerce_to_float32(self):
        scaler = StandardScaler(mean=[0.0], std=[1.0])
        self.assertEqual(scaler.mean.dtype, np.dtype("float32"))
        self.assertEqual(scaler.std.dtype, np.dtype("float32"))

    def test_transform_with_dtype_none_does_not_promote_to_float64(self):
        """astype(None) silently yields float64, so the cast stays guarded."""
        scaler = StandardScaler(mean=[0.0], std=[1.0])
        self.assertIsNone(scaler.dtype)

        X = self.xp.ones((4, 1), dtype=self.xp.float32)
        self.assertEqual(scaler.transform(X).dtype, np.dtype("float32"))

    # ---- axis coercion ----

    def test_list_axis_is_coerced_to_tuple(self):
        # A list axis would be forwarded straight into xp.mean/std, which only
        # accepts int or tuple.
        self.assertEqual(StandardScaler(axis=[0, 1, 2]).axis, (0, 1, 2))

    def test_list_axis_fits_a_four_dimensional_batch(self):
        scaler = StandardScaler(axis=[0, 1, 2])
        scaler.fit(self.xp.zeros((2, 3, 3, 4), dtype=self.xp.float32))
        self.assertEqual(scaler.mean.shape, (1, 1, 1, 4))

    def test_axis_round_trips_through_config_as_a_tuple(self):
        restored = StandardScaler.from_config(
            StandardScaler(axis=(0, 1, 2)).get_config()
        )
        self.assertEqual(restored.axis, (0, 1, 2))

    # ---- dtype pin serialization ----

    def test_dtype_pin_survives_a_config_round_trip(self):
        restored = StandardScaler.from_config(
            StandardScaler(dtype="float32").get_config()
        )
        self.assertTrue(restored._dtype_pinned)

    def test_inferred_dtype_does_not_come_back_pinned(self):
        """A dtype fit() merely inferred must stay overridable after a reload."""
        scaler = StandardScaler(axis=0).fit(
            self.xp.ones((4, 3), dtype=self.xp.float32)
        )
        restored = StandardScaler.from_config(scaler.get_config())

        self.assertFalse(restored._dtype_pinned)
        self.assertEqual(restored.dtype, np.dtype("float32"))


base_case.register_test_suites(globals(), TestStandardScalerTransform)