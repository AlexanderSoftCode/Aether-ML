
import tests.base_case as base_case
from aether.preprocessing.transforms import (
    Compose,
    Rescale,
    StandardScaler,
    ToTensor,
    deserialize,
)


def _entry(transform):
    """Builds the {class_name, config} shape Model.save() writes."""
    return {"class_name": type(transform).__name__, "config": transform.get_config()}


class TestPreprocessorDeserializeBase(base_case.AetherBaseTestCase):
    __test__ = False

    # ---- Happy paths ----

    def test_none_entry_returns_none(self):
        """A model saved with no pipeline writes a null entry."""
        self.assertIsNone(deserialize(None))

    def test_empty_entry_returns_none(self):
        self.assertIsNone(deserialize({}))

    def test_leaf_transform_round_trips(self):
        restored = deserialize(_entry(Rescale(factor=0.25)))
        self.assertIsInstance(restored, Rescale)
        self.assertEqual(restored.factor, 0.25)

    def test_to_tensor_round_trips_with_device_and_dtype(self):
        restored = deserialize(
            _entry(ToTensor(dtype="float32", target_device=self.backend_name))
        )
        self.assertIsInstance(restored, ToTensor)
        self.assertEqual(restored.dtype, "float32")
        self.assertEqual(restored.target_device, self.backend_name)

    def test_compose_round_trips_members_in_order(self):
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            StandardScaler(axis=0),
        ])
        restored = deserialize(_entry(pipeline))

        self.assertIsInstance(restored, Compose)
        self.assertEqual(
            [type(t).__name__ for t in restored.transforms],
            ["ToTensor", "Rescale", "StandardScaler"],
        )

    def test_nested_compose_round_trips_recursively(self):
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            Compose([Rescale(factor=1.0 / 255.0), StandardScaler(axis=0)]),
        ])
        restored = deserialize(_entry(pipeline))

        self.assertIsInstance(restored.transforms[1], Compose)
        self.assertIsInstance(restored.transforms[1].transforms[1], StandardScaler)

    def test_fitted_scaler_inside_compose_restores_usable_statistics(self):
        X = self.xp.arange(12, dtype=self.xp.float32).reshape(4, 3)
        pipeline = Compose([StandardScaler(axis=0)]).fit(X)
        restored = deserialize(_entry(pipeline))

        self.assertTrue(restored.is_fitted)
        self.assertTrue(self.xp.allclose(restored(X), pipeline(X), atol=1e-5))

    # ---- Error paths ----

    def test_unknown_class_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            deserialize({"class_name": "HypotheticalWhiteningTransform", "config": {}})
        self.assertIn("HypotheticalWhiteningTransform", str(ctx.exception))

    def test_base_preprocess_class_raises_type_error(self):
        with self.assertRaises(TypeError) as ctx:
            deserialize({"class_name": "Preprocess", "config": {}})
        self.assertIn("Preprocess", str(ctx.exception))

    def test_non_preprocess_module_global_raises_type_error(self):
        # `to_tensor` is a module-level function, not a Preprocess subclass --
        # resolving by name alone would happily call it.
        with self.assertRaises(TypeError):
            deserialize({"class_name": "to_tensor", "config": {}})

    def test_unknown_class_nested_in_compose_raises_value_error(self):
        entry = {
            "class_name": "Compose",
            "config": {
                "transforms": [
                    {"class_name": "HypotheticalWhiteningTransform", "config": {}}
                ]
            },
        }
        with self.assertRaises(ValueError):
            deserialize(entry)

    def test_stale_config_signature_raises_type_error(self):
        with self.assertRaises(TypeError) as ctx:
            deserialize({"class_name": "Rescale", "config": {"gain": 2.0}})
        self.assertIn("Rescale", str(ctx.exception))


base_case.register_test_suites(globals(), TestPreprocessorDeserializeBase)
