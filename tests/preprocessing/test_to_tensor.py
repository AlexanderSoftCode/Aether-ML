import json
import warnings

import numpy as np

import aether.config as config
import aether.preprocessing
import tests.base_case as base_case

from aether.preprocessing.transforms import ToTensor, to_tensor
from aether.preprocessing._utils import (
    validate_dtype,
    is_dtype_like,
    parse_inputs,
    resolve_dtypes,
    convert_single_tensor,
)


class TestToTensorTransform(base_case.AetherBaseTestCase):

    # ---- Helper Function Tests (_utils.py) ----------------------

    def test_validate_dtype_valid_and_invalid(self):
        """Verify valid floating dtypes pass and invalid dtypes raise ValueError."""
        self.assertIsNone(validate_dtype(None))
        self.assertEqual(validate_dtype('float32'), 'float32')
        self.assertEqual(validate_dtype(np.float32), np.float32)

        with self.assertRaises(ValueError):
            validate_dtype('int32')  # Only allowed float precision types

        with self.assertRaises(ValueError):
            validate_dtype('unsupported_type')

    def test_validate_dtype_float64_warning(self):
        """Verify float64 raises a performance degradation UserWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with self.assertWarns(UserWarning):
                validate_dtype('float64')

    def test_is_dtype_like(self):
        """Verify identification of valid scalar and sequence dtypes."""
        self.assertTrue(is_dtype_like('float32'))
        self.assertTrue(is_dtype_like(('float32', 'float16')))
        self.assertTrue(is_dtype_like([np.float32, None]))
        self.assertFalse(is_dtype_like('not_a_dtype'))

        self.assertFalse(is_dtype_like('not_a_dtype'))
        self.assertFalse(is_dtype_like(['float32', 'invalid_dtype']))

    def test_parse_inputs(self):
        """Verify argument decomposition for trailing positional dtypes."""
        arr = [1.0, 2.0, 3.0]
        
        parsed_args, dt = parse_inputs((arr, 'float32'), kw_dtype=None)
        self.assertEqual(len(parsed_args), 1)
        self.assertEqual(dt, 'float32')

        parsed_args, dt = parse_inputs((arr,), kw_dtype='float16')
        self.assertEqual(dt, 'float16')

    def test_resolve_dtypes(self):
        """Verify dtype broadcasting and 1:1 positional mapping."""
        self.assertEqual(resolve_dtypes('float32', 2), ['float32', 'float32'])
        
        self.assertEqual(resolve_dtypes(('float32', 'float16'), 2), ['float32', 'float16'])

        with self.assertRaises(ValueError):
            resolve_dtypes(('float32', 'float16'), 3)

    def test_convert_single_tensor_updated_integer_preservation(self):
        """Verify convert_single_tensor properly handles integer bypass logic."""
        arr = np.array([1, 2, 3], dtype=np.uint8)

        # When target_dtype is None and preserve_integers=True -> keep integer dtype
        res_kept = convert_single_tensor(
            arr, target_dtype=None, target_device=self.backend_name, preserve_integers=True
        )
        self.assertTrue(self.xp.issubdtype(res_kept.dtype, self.xp.integer))

        res_casted = convert_single_tensor(
            arr, target_dtype='float32', target_device=self.backend_name, preserve_integers=True
        )
        self.assertEqual(str(res_casted.dtype), 'float32')

    # ---- Functional API Tests (to_tensor) -----------------------------

    def test_to_tensor_single_array_conversion(self):
        """Verify single array conversion to target device and precision."""
        arr = [1, 2, 3]
        res = to_tensor(arr, target_device=self.backend_name, dtype='float32')
        
        self.assertIsInstance(res, self.xp.ndarray)
        self.assertEqual(str(res.dtype), 'float32')

    def test_to_tensor_multi_array_tuple_unpacking(self):
        """Verify multiple positional array inputs return a tuple with matching types."""
        X = [[1, 2], [3, 4]]
        y = [0, 1]

        X_res, y_res = to_tensor(
            X, y, 
            target_device=self.backend_name, 
            dtype=('float32', None), 
            preserve_integers=True
        )

        self.assertIsInstance(X_res, self.xp.ndarray)
        self.assertEqual(str(X_res.dtype), 'float32')
        
        self.assertIsInstance(y_res, self.xp.ndarray)
        self.assertTrue(self.xp.issubdtype(y_res.dtype, self.xp.integer))

    # ---- Class Pipeline Wrapper Tests (ToTensor) --------------

    def test_totensor_class_instance(self):
        """Verify ToTensor instance correctly delegates config arguments via __call__."""
        transform = ToTensor(
            dtype='float32',
            preserve_integers=True,
            target_device=self.backend_name
        )

        raw_data = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        tensor_data = transform(raw_data)

        self.assertIsInstance(tensor_data, self.xp.ndarray)
        self.assertEqual(str(tensor_data.dtype), 'float32')

    # ---- Model Dispatch Hook Tests (_compile_for_device / _apply_precision) ----

    def test_compile_for_device_overrides_explicit_target_device(self):
        """Model.to() owns the device target, so it wins over a user-set value."""
        transform = ToTensor(target_device="numpy")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            transform._compile_for_device(self.backend_name)

        self.assertEqual(transform.target_device, self.backend_name)

    def test_compile_for_device_fills_unset_target_device(self):
        transform = ToTensor()
        transform._compile_for_device(self.backend_name)
        self.assertEqual(transform.target_device, self.backend_name)

    def test_compile_for_device_warns_when_overriding_a_pinned_device(self):
        """model.to() still wins, but a user who explicitly pinned target_device
        should be told their pin is being discarded, not have it happen silently."""
        transform = ToTensor(target_device="numpy")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform._compile_for_device("cupy")

        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertIn("overrides it", str(caught[0].message))

        # The override still happens -- model.to() remains the source of truth.
        self.assertEqual(transform.target_device, "cupy")

    def test_compile_for_device_no_warning_when_unpinned(self):
        """No pin was ever set, so there's nothing to silently discard."""
        transform = ToTensor()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform._compile_for_device("cupy")

        self.assertEqual(caught, [])

    def test_compile_for_device_no_warning_when_new_device_matches_pin(self):
        """The pinned value isn't actually being overridden, so no warning is due."""
        transform = ToTensor(target_device="numpy")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            transform._compile_for_device("numpy")

        self.assertEqual(caught, [])

    def test_apply_precision_fills_unset_dtype(self):
        transform = ToTensor()
        transform._apply_precision(config.DTypePolicy(compute_dtype="float32"))
        self.assertEqual(transform.dtype, "float32")

    def test_apply_precision_leaves_explicit_dtype_pinned(self):
        """An explicitly constructed dtype is the user's pin and outranks the policy."""
        transform = ToTensor(dtype="float32")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*NumPy float16 is emulated.*", category=UserWarning
            )
            transform._apply_precision(config.DTypePolicy(compute_dtype="float16"))

        self.assertEqual(transform.dtype, "float32")

    def test_apply_precision_noop_when_policy_has_no_compute_dtype(self):
        transform = ToTensor()
        transform._apply_precision(config.DTypePolicy(compute_dtype=None))
        self.assertIsNone(transform.dtype)

    def test_apply_precision_stores_a_json_safe_string(self):
        transform = ToTensor()
        transform._apply_precision(config.DTypePolicy(compute_dtype="float32"))
        json.dumps(transform.get_config())

    def test_get_config_stringifies_a_dtype_object(self):
        cfg = ToTensor(dtype=np.float32, target_device="numpy").get_config()
        self.assertEqual(
            cfg,
            {"dtype": "float32", "preserve_integers": True, "target_device": "numpy", "dtype_pinned": True, "device_pinned": True},
        )

    # ---- Regression tests: dtype pin and multiple set_precision calls ----

    def test_apply_precision_overwrites_unpinned_dtype(self):
        """Unpinned ToTensor: successive _apply_precision calls overwrite dtype."""
        transform = ToTensor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            transform._apply_precision(config.DTypePolicy(compute_dtype="float16"))
        self.assertEqual(transform.dtype, "float16")

        transform._apply_precision(config.DTypePolicy(compute_dtype="float32"))
        self.assertEqual(transform.dtype, "float32")

    def test_apply_precision_preserves_pinned_dtype(self):
        """Pinned ToTensor(dtype='float32'): _apply_precision leaves dtype alone."""
        transform = ToTensor(dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            transform._apply_precision(config.DTypePolicy(compute_dtype="float16"))
        self.assertEqual(transform.dtype, "float32")

    def test_roundtrip_unpinned_but_filled_dtype(self):
        """Unpinned ToTensor with filled dtype: roundtrip via deserialize preserves pin state."""
        transform = ToTensor()
        transform._apply_precision(config.DTypePolicy(compute_dtype="float32"))

        cfg = transform.get_config()
        restored = aether.preprocessing.deserialize({"class_name": "ToTensor", "config": cfg})

        # After roundtrip, it should still be unpinned
        self.assertFalse(restored._dtype_pinned)
        self.assertEqual(restored.dtype, "float32")

        # Verify another set_precision call will overwrite it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            restored._apply_precision(config.DTypePolicy(compute_dtype="float16"))
        self.assertEqual(restored.dtype, "float16")

    def test_roundtrip_pinned_dtype(self):
        """Pinned ToTensor: roundtrip via deserialize preserves pin state."""
        transform = ToTensor(dtype="float32")

        cfg = transform.get_config()
        restored = aether.preprocessing.deserialize({"class_name": "ToTensor", "config": cfg})

        # After roundtrip, it should still be pinned
        self.assertTrue(restored._dtype_pinned)
        self.assertEqual(restored.dtype, "float32")

        # Verify set_precision is ignored
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            restored._apply_precision(config.DTypePolicy(compute_dtype="float16"))
        self.assertEqual(restored.dtype, "float32")

    def test_roundtrip_legacy_config_without_dtype_pinned_key(self):
        """Legacy config without dtype_pinned key: defaults to treating non-None dtype as pinned."""
        # Simulate old config (before dtype_pinned key was added)
        legacy_cfg = {
            "dtype": "float32",
            "preserve_integers": True,
            "target_device": None,
        }

        restored = aether.preprocessing.deserialize({"class_name": "ToTensor", "config": legacy_cfg})

        # Should come back pinned (since dtype is not None and key is missing)
        self.assertTrue(restored._dtype_pinned)
        self.assertEqual(restored.dtype, "float32")

        # Verify set_precision is ignored (pinned behavior)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            restored._apply_precision(config.DTypePolicy(compute_dtype="float16"))
        self.assertEqual(restored.dtype, "float32")

    def test_roundtrip_legacy_config_with_none_dtype(self):
        """Legacy config without dtype_pinned but dtype=None: comes back unpinned."""
        legacy_cfg = {
            "dtype": None,
            "preserve_integers": True,
            "target_device": None,
        }

        restored = aether.preprocessing.deserialize({"class_name": "ToTensor", "config": legacy_cfg})

        # Should come back unpinned (since dtype is None)
        self.assertFalse(restored._dtype_pinned)
        self.assertIsNone(restored.dtype)

        # Verify set_precision works
        restored._apply_precision(config.DTypePolicy(compute_dtype="float32"))
        self.assertEqual(restored.dtype, "float32")

    # ---- Regression tests: device pin ----

    def test_roundtrip_unpinned_but_filled_target_device(self):
        """Unpinned ToTensor with filled target_device: roundtrip via deserialize preserves pin state."""
        transform = ToTensor()
        transform._compile_for_device("numpy")

        cfg = transform.get_config()
        restored = aether.preprocessing.deserialize({"class_name": "ToTensor", "config": cfg})

        # After roundtrip, it should still be unpinned
        self.assertFalse(restored._device_pinned)
        self.assertEqual(restored.target_device, "numpy")

        # Verify _compile_for_device with a different device emits no warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored._compile_for_device("cupy")

        self.assertEqual(caught, [])
        self.assertEqual(restored.target_device, "cupy")

    def test_roundtrip_pinned_target_device(self):
        """Pinned ToTensor: roundtrip via deserialize preserves pin state."""
        transform = ToTensor(target_device="numpy")

        cfg = transform.get_config()
        restored = aether.preprocessing.deserialize({"class_name": "ToTensor", "config": cfg})

        # After roundtrip, it should still be pinned
        self.assertTrue(restored._device_pinned)
        self.assertEqual(restored.target_device, "numpy")

        # Verify _compile_for_device with a different device emits a warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored._compile_for_device("cupy")

        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertIn("overrides it", str(caught[0].message))

    def test_roundtrip_legacy_config_without_device_pinned_key(self):
        """Legacy config without device_pinned key but with target_device: comes back pinned."""
        # Simulate old config (before device_pinned key was added)
        legacy_cfg = {
            "dtype": None,
            "preserve_integers": True,
            "target_device": "numpy",
        }

        restored = aether.preprocessing.deserialize({"class_name": "ToTensor", "config": legacy_cfg})

        # Should come back pinned (since target_device is not None and key is missing)
        self.assertTrue(restored._device_pinned)
        self.assertEqual(restored.target_device, "numpy")

        # Verify _compile_for_device with a different device emits a warning (pinned behavior)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored._compile_for_device("cupy")

        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))

    def test_compile_for_device_clears_pin_after_warning(self):
        """Regression test: after pinned device is overridden, subsequent overrides emit no warning."""
        t = ToTensor(target_device='cupy')
        self.assertTrue(t._device_pinned)

        # First override: should warn and clear the pin
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t._compile_for_device('numpy')

        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertFalse(t._device_pinned)
        self.assertEqual(t.target_device, 'numpy')

        # Second override: no warning (pin is already cleared)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t._compile_for_device('cupy')

        self.assertEqual(len(caught), 0)
        self.assertEqual(t.target_device, 'cupy')

        # Third override: still no warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t._compile_for_device('numpy')

        self.assertEqual(len(caught), 0)
        self.assertEqual(t.target_device, 'numpy')

        # get_config should reflect unpinned state
        self.assertFalse(t.get_config()["device_pinned"])


base_case.register_test_suites(globals(), TestToTensorTransform)