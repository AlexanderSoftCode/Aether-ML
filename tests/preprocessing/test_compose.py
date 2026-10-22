import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.preprocessing.transforms import (
    Compose,
    Preprocess,
    Rescale,
    StandardScaler,
    ToTensor,
)


class TestComposePipelineTransform(base_case.AetherBaseTestCase):

    def test_compose_single_array_totensor_and_scaler_pipeline(self):
        """Verify end-to-end pipeline execution with a pre-fitted scaler."""
        X_train_raw = np.random.randint(0, 256, size=(100, 32, 32, 3), dtype=np.uint8)

        X_train_tensor = config.to_device(X_train_raw, target=self.backend_name).astype(self.xp.float32) / 255.0
        scaler = StandardScaler().fit(X_train_tensor)

        pipeline = Compose(transforms=[
            ToTensor(dtype='float32', target_device=self.backend_name),
            scaler
        ])

        X_test_raw = np.random.randint(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
        out_tensor = pipeline(X_test_raw)

        self.assertIsInstance(out_tensor, self.xp.ndarray)
        self.assertEqual(out_tensor.shape, (20, 32, 32, 3))
        self.assertEqual(str(out_tensor.dtype), 'float32')

    def test_compose_fit_with_unfitted_scaler(self):
        """Verify Compose.fit propagates intermediate transformations and fits unfitted scalers."""
        raw_train = np.array([[0, 128], [255, 255]], dtype=np.uint8)
        raw_test = np.array([[128, 255]], dtype=np.uint8)

        scaler = StandardScaler()
        self.assertIsNone(scaler.mean, "Scaler should be unfitted initially")

        pipeline = Compose(transforms=[
            ToTensor(dtype='float32', target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            scaler
        ])

        returned_pipeline = pipeline.fit(raw_train)
        self.assertIs(returned_pipeline, pipeline)

        self.assertIsNotNone(scaler.mean)
        self.assertIsNotNone(scaler.std)

        expected_mean = float(self.xp.mean(self.xp.array([[0.0, 128.0 / 255.0], [1.0, 1.0]], dtype=self.xp.float32)))
        self.assertAlmostEqual(float(scaler.mean.item()), expected_mean, places=4)

        train_out = pipeline(raw_train)
        test_out = pipeline(raw_test)

        self.assertIsInstance(train_out, self.xp.ndarray)
        self.assertIsInstance(test_out, self.xp.ndarray)
        self.assertAlmostEqual(float(self.xp.mean(train_out)), 0.0, places=4)

    def test_compose_fit_transform_equivalence(self):
        """Verify Compose.fit_transform(X) yields identical results to Compose.fit(X)(X)."""
        raw_train = np.random.randint(0, 256, size=(50, 8), dtype=np.uint8)

        pipeline_fit_transform = Compose(transforms=[
            ToTensor(dtype='float32', target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            StandardScaler()
        ])
        out_fit_transform = pipeline_fit_transform.fit_transform(raw_train)

        pipeline_fit = Compose(transforms=[
            ToTensor(dtype='float32', target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            StandardScaler()
        ])
        pipeline_fit.fit(raw_train)
        out_fit_call = pipeline_fit(raw_train)

        self.assertTrue(self.xp.allclose(out_fit_transform, out_fit_call, atol=1e-5))
        self.assertEqual(str(out_fit_transform.dtype), 'float32')

    def test_compose_sequential_execution_order(self):
        """Verify that Compose executes transforms sequentially in exact list order."""
        execution_order = []

        class StepOne(Preprocess):
            def transform(self, x):
                execution_order.append(1)
                return x + 10

        class StepTwo(Preprocess):
            def transform(self, x):
                execution_order.append(2)
                return x * 2

        pipeline = Compose(transforms=[StepOne(), StepTwo()])
        result = pipeline(5)

        self.assertEqual(execution_order, [1, 2])
        self.assertEqual(result, (5 + 10) * 2)

    def test_compose_multiple_inputs_tuple_unpacking(self):
        """Verify Compose correctly handles multiple positional inputs across transforms."""
        X_raw = np.ones((10, 4), dtype=np.float64)
        y_raw = np.zeros((10,), dtype=np.int64)

        class DoubleBoth(Preprocess):
            def transform(self, X, y):
                return X * 2, y + 1

        pipeline = Compose(transforms=[
            ToTensor(dtype='float32', preserve_integers=True, target_device=self.backend_name),
            DoubleBoth()
        ])

        X_out, y_out = pipeline(X_raw, y_raw)

        self.assertIsInstance(X_out, self.xp.ndarray)
        self.assertIsInstance(y_out, self.xp.ndarray)
        self.assertEqual(str(X_out.dtype), 'float32')
        self.assertTrue(self.xp.issubdtype(y_out.dtype, self.xp.integer))
        self.assertEqual(float(X_out[0, 0]), 2.0)
        self.assertEqual(int(y_out[0]), 1)

    def test_compose_empty_transforms_returns_input(self):
        """Verify empty transform list returns input data untouched."""
        pipeline = Compose(transforms=[])
        sample_data = self.xp.array([1, 2, 3])

        out = pipeline(sample_data)
        self.assertTrue(self.xp.array_equal(out, sample_data))

    def test_compose_fit_empty_pipeline_returns_self(self):
        """Verify calling .fit() and .fit_transform() on an empty pipeline does not error."""
        pipeline = Compose(transforms=[])
        sample_data = self.xp.array([1.0, 2.0, 3.0])

        fitted = pipeline.fit(sample_data)
        self.assertIs(fitted, pipeline)

        out = pipeline.fit_transform(sample_data)
        self.assertTrue(self.xp.array_equal(out, sample_data))

    def test_compose_single_input_with_rescale(self):
        """Verify Compose handles ToTensor -> Rescale pipeline on sRGB image arrays."""
        raw_images = np.full((10, 32, 32, 3), 255, dtype=np.uint8)

        pipeline = Compose(transforms=[
            ToTensor(dtype='float32', target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0)
        ])

        res = pipeline(raw_images)

        self.assertIsInstance(res, self.xp.ndarray)
        self.assertEqual(str(res.dtype), 'float32')
        self.assertTrue(self.xp.allclose(res, 1.0))

    def test_compose_rescale_chained_with_scaler(self):
        """Verify Compose correctly chains ToTensor -> Rescale -> pre-fitted StandardScaler."""
        raw_images = np.array([[0, 128], [255, 255]], dtype=np.uint8)

        scaled_data = raw_images.astype(np.float32) / 255.0
        scaler = StandardScaler().fit(scaled_data)

        pipeline = Compose(transforms=[
            ToTensor(dtype='float32', target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            scaler
        ])

        res = pipeline(raw_images)

        self.assertIsInstance(res, self.xp.ndarray)
        self.assertEqual(str(res.dtype), 'float32')
        self.assertAlmostEqual(float(self.xp.mean(res)), 0.0, places=4)

    def test_compose_multi_input_rescale_preserves_labels(self):
        """Verify Rescale in a multi-input pipeline scales float features but ignores integer targets."""
        X_raw = np.array([[0, 255], [127, 255]], dtype=np.uint8)
        y_raw = np.array([0, 1], dtype=np.int64)

        pipeline = Compose(transforms=[
            ToTensor(dtype='float32', preserve_integers=True, target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0)
        ])

        X_out, y_out = pipeline(X_raw, y_raw)

        self.assertEqual(str(X_out.dtype), 'float32')
        self.assertAlmostEqual(float(X_out[0, 1]), 1.0, places=5)
        self.assertTrue(self.xp.issubdtype(y_out.dtype, self.xp.integer))
        self.assertEqual(int(y_out[1]), 1)


    # ---- Contract, construction and nesting ----

    def test_compose_is_a_preprocess(self):
        """Compose is dispatched like any other transform, so it must be one."""
        self.assertIsInstance(Compose(transforms=[]), Preprocess)

    def test_compose_transform_and_call_agree(self):
        X = self.xp.array([1.0, 2.0, 3.0], dtype=self.xp.float32)
        pipeline = Compose(transforms=[Rescale(factor=2.0)])
        self.assertTrue(self.xp.allclose(pipeline.transform(X), pipeline(X)))

    def test_compose_rejects_non_preprocess_member_naming_index(self):
        """Members are dispatched unguarded, so the rejection has to name the slot."""
        with self.assertRaises(TypeError) as ctx:
            Compose(transforms=[ToTensor(), object(), Rescale()])

        message = str(ctx.exception)
        self.assertIn("transforms[1]", message)
        self.assertIn("object", message)

    def test_compose_rejects_bare_callable_member(self):
        # Callable is not enough: _compile_for_device/_apply_precision/get_config
        # are all dispatched on members without a hasattr guard.
        with self.assertRaises(TypeError):
            Compose(transforms=[lambda X: X])

    def test_nested_compose_executes_in_flattened_order(self):
        order = []

        class Step(Preprocess):
            def __init__(self, tag):
                self.tag = tag

            def transform(self, X):
                order.append(self.tag)
                return X

        pipeline = Compose(transforms=[
            Step("a"),
            Compose(transforms=[Step("b"), Step("c")]),
            Step("d"),
        ])
        pipeline(self.xp.zeros((2, 2), dtype=self.xp.float32))

        self.assertEqual(order, ["a", "b", "c", "d"])

    def test_nested_compose_transforms_values_in_order(self):
        pipeline = Compose(transforms=[
            Rescale(factor=2.0),
            Compose(transforms=[Rescale(factor=3.0)]),
        ])
        X = self.xp.array([1.0], dtype=self.xp.float32)
        self.assertAlmostEqual(float(pipeline(X)[0]), 6.0, places=5)

    # ---- is_fitted aggregation ----

    def test_is_fitted_aggregates_members(self):
        scaler = StandardScaler()
        pipeline = Compose(transforms=[
            ToTensor(dtype="float32", target_device=self.backend_name),
            scaler,
        ])
        self.assertFalse(pipeline.is_fitted)

        scaler.fit(self.xp.ones((4, 3), dtype=self.xp.float32))
        self.assertTrue(pipeline.is_fitted)

    def test_is_fitted_true_for_stateless_and_empty_pipelines(self):
        self.assertTrue(Compose(transforms=[]).is_fitted)
        self.assertTrue(Compose(transforms=[ToTensor(), Rescale()]).is_fitted)

    def test_is_fitted_aggregates_through_nesting(self):
        scaler = StandardScaler()
        pipeline = Compose(transforms=[Compose(transforms=[scaler])])
        self.assertFalse(pipeline.is_fitted)

        scaler.fit(self.xp.ones((4, 3), dtype=self.xp.float32))
        self.assertTrue(pipeline.is_fitted)


base_case.register_test_suites(globals(), TestComposePipelineTransform)