import warnings

import numpy as np

import aether.config as config
import tests.base_case as base_case

from aether._utils import NullPreprocessor
from aether.layers import Dense, ReLU
from aether.losses import CategoricalCrossEntropy
from aether.metrics import CategoricalAccuracy
from aether.model import Model
from aether.optimizers import Adam
from aether.preprocessing.transforms import (
    Compose,
    Preprocess,
    Rescale,
    StandardScaler,
    ToTensor,
)


class SpyCountingTransform(Preprocess):
    """Records every transform() invocation and the batch length it saw."""

    def __init__(self):
        self.calls = 0
        self.batch_lengths = []

    def transform(self, X):
        self.calls += 1
        self.batch_lengths.append(len(X))
        return X


class TestModelPreprocessorBase(base_case.AetherBaseTestCase):
    __test__ = False

    NUM_SAMPLES = 32
    NUM_FEATURES = 6
    NUM_CLASSES = 3

    def setUp(self):
        super().setUp()
        np.random.seed(11)
        self.raw_X = np.random.randint(
            0, 256, size=(self.NUM_SAMPLES, self.NUM_FEATURES), dtype=np.uint8
        )
        self.raw_y = np.random.randint(
            0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)
        ).astype(np.int32)
        # A visibly different distribution, for proving a refit did or did not run.
        self.shifted_X = np.clip(self.raw_X.astype(np.int16) - 120, 0, 255).astype(
            np.uint8
        )

    def make_pipeline(self, fitted=True):
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            StandardScaler(),
        ])
        return pipeline.fit(self.raw_X) if fitted else pipeline

    def build_model(self, preprocessor=None, finalize=True):
        model = Model()
        model.manual_seed(3)
        model.add(Dense(self.NUM_FEATURES, 8))
        model.add(ReLU())
        model.add(Dense(8, self.NUM_CLASSES))
        model.configure(
            loss=CategoricalCrossEntropy(),
            optimizer=Adam(lr=0.01),
            accuracy=CategoricalAccuracy(),
            preprocessor=preprocessor,
        )
        model.to(self.backend_name)
        if finalize:
            model.finalize((self.NUM_FEATURES,))
        return model

    def set_precision(self, model, compute_dtype="float16"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*NumPy float16 is emulated.*", category=UserWarning
            )
            model.set_precision(compute_dtype)

    # ---- configure() ----

    def test_configure_rejects_non_preprocess(self):
        with self.assertRaises(TypeError) as ctx:
            Model().configure(preprocessor=object())
        self.assertIn("Preprocess", str(ctx.exception))

    def test_configure_accepts_a_preprocessor_alone(self):
        """The all-None guard was widened for exactly this call shape."""
        pipeline = self.make_pipeline()
        model = Model()
        model.configure(preprocessor=pipeline)
        self.assertIs(model.preprocessor, pipeline)

    def test_configure_with_no_components_still_raises(self):
        with self.assertRaises(ValueError):
            Model().configure()

    # ---- finalize() and the null object ----

    def test_finalize_installs_null_preprocessor_when_none_attached(self):
        model = self.build_model()
        self.assertIsInstance(model.preprocessor, NullPreprocessor)
        self.assertFalse(model._has_pipeline())

    def test_finalize_leaves_an_attached_pipeline_in_place(self):
        pipeline = self.make_pipeline()
        model = self.build_model(preprocessor=pipeline)
        self.assertIs(model.preprocessor, pipeline)
        self.assertTrue(model._has_pipeline())

    def test_null_preprocessor_transform_is_a_pure_identity(self):
        """It must not migrate devices -- every pipeline-free model owns one."""
        host_array = np.ones((2, 2), dtype=np.float32)
        self.assertIs(NullPreprocessor().transform(host_array), host_array)

    def test_null_preprocessor_does_not_migrate_for_a_device_model(self):
        model = self.build_model()
        host_array = np.ones((2, 2), dtype=np.float32)
        self.assertIs(model.preprocessor.transform(host_array), host_array)

    # ---- _sync_device ----

    def test_to_migrates_standard_scaler_statistics(self):
        scaler = StandardScaler(axis=0).fit(
            np.arange(12, dtype=np.float32).reshape(4, 3)
        )
        model = Model()
        model.add(Dense(3, 2))
        model.configure(preprocessor=scaler)
        model.to(self.backend_name)

        self.assertIs(config.get_array_module(scaler.mean), self.xp)
        self.assertIs(config.get_array_module(scaler.std), self.xp)

    def test_to_retargets_to_tensor_inside_the_pipeline(self):
        pipeline = Compose([ToTensor(target_device="numpy")])
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 2))
        model.configure(preprocessor=pipeline)

        with warnings.catch_warnings():
            # On the cupy backend, the pinned "numpy" target mismatches
            # model.to("cupy") and triggers the pin-override warning -- not
            # what this test is checking, so it's silenced here.
            warnings.simplefilter("ignore", category=UserWarning)
            model.to(self.backend_name)

        self.assertEqual(pipeline.transforms[0].target_device, self.backend_name)

    # ---- set_precision() ----

    def test_set_precision_reaches_the_pipeline(self):
        pipeline = Compose([ToTensor(target_device=self.backend_name)])
        model = self.build_model(preprocessor=pipeline, finalize=False)
        self.set_precision(model)

        self.assertEqual(pipeline.transforms[0].dtype, "float16")

    def test_set_precision_before_configure_still_lands_via_finalize(self):
        model = Model()
        model.manual_seed(3)
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        self.set_precision(model)

        pipeline = Compose([ToTensor(target_device=self.backend_name)])
        model.configure(preprocessor=pipeline)
        model.to(self.backend_name)
        model.finalize((self.NUM_FEATURES,))

        self.assertEqual(pipeline.transforms[0].dtype, "float16")

    def test_finalize_without_a_policy_leaves_the_pipeline_alone(self):
        pipeline = Compose([ToTensor(target_device=self.backend_name)])
        self.build_model(preprocessor=pipeline)
        self.assertIsNone(pipeline.transforms[0].dtype)

    # ---- Device contract ----

    def test_pipeline_accepts_raw_host_input_for_train(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        model.train(
            self.raw_X, self.raw_y, epochs=1, batch_size=8, verbose=0, print_every=0
        )

    def test_pipeline_accepts_raw_host_input_for_evaluate(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        loss, acc = model.evaluate(self.raw_X, self.raw_y, batch_size=8, verbose=0)

        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(acc, 0.0)

    def test_without_a_pipeline_the_strict_device_guard_is_unchanged(self):
        """Backward-compatibility contract: no pipeline means no implicit migration."""
        if "cupy" not in base_case.BACKENDS_TO_TEST:
            self.skipTest("Both backends are required to build a cross-device input.")

        other = "numpy" if self.backend_name == "cupy" else "cupy"
        X = config.to_device(self.raw_X.astype(np.float32), target=other)
        y = config.to_device(self.raw_y, target=other)

        model = self.build_model()
        with self.assertRaises(TypeError) as ctx:
            model.train(X, y, epochs=1, verbose=0, print_every=0)
        self.assertIn("Device mismatch", str(ctx.exception))

        with self.assertRaises(TypeError):
            model.evaluate(X, y, verbose=0)

    def test_assert_pipeline_device_raises_its_targeted_error(self):
        if "cupy" not in base_case.BACKENDS_TO_TEST:
            self.skipTest("Both backends are required to build a cross-device pipeline.")

        other = "numpy" if self.backend_name == "cupy" else "cupy"
        # A stateless pipeline on purpose: the probe is about placement alone.
        pipeline = Compose([ToTensor(dtype="float32", target_device=self.backend_name)])
        model = self.build_model(preprocessor=pipeline)
        # Retarget after finalize: model.to() would otherwise overwrite it.
        pipeline.transforms[0].target_device = other

        with self.assertRaises(TypeError) as ctx:
            model.evaluate(self.raw_X, self.raw_y, verbose=0)

        message = str(ctx.exception)
        self.assertIn("Preprocessing pipeline device mismatch", message)
        self.assertIn(self.backend_name, message)
        self.assertIn(other, message)

    def test_assert_pipeline_device_probes_only_one_sample(self):
        spy = SpyCountingTransform()
        model = self.build_model(
            preprocessor=Compose([
                ToTensor(dtype="float32", target_device=self.backend_name),
                spy,
            ])
        )
        model.evaluate(self.raw_X, self.raw_y, verbose=0)

        self.assertEqual(spy.batch_lengths[0], 1)

    # ---- fit_preprocessor tri-state ----

    def test_fit_preprocessor_none_fits_only_when_unfitted(self):
        pipeline = self.make_pipeline(fitted=False)
        model = self.build_model(preprocessor=pipeline)
        self.assertFalse(pipeline.is_fitted)

        model.train(
            self.raw_X, self.raw_y, epochs=1, batch_size=8, verbose=0, print_every=0
        )
        self.assertTrue(pipeline.is_fitted)

        first_mean = float(pipeline.transforms[-1].mean.ravel()[0])
        model.train(
            self.shifted_X, self.raw_y, epochs=1, batch_size=8, verbose=0, print_every=0
        )
        self.assertAlmostEqual(
            float(pipeline.transforms[-1].mean.ravel()[0]), first_mean, places=6
        )

    def test_fit_preprocessor_true_always_refits(self):
        pipeline = self.make_pipeline()
        model = self.build_model(preprocessor=pipeline)
        before = float(pipeline.transforms[-1].mean.ravel()[0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            model.train(
                self.shifted_X,
                self.raw_y,
                epochs=1,
                batch_size=8,
                verbose=0,
                print_every=0,
                fit_preprocessor=True,
            )
        self.assertNotAlmostEqual(
            float(pipeline.transforms[-1].mean.ravel()[0]), before, places=3
        )

    def test_fit_preprocessor_true_on_an_already_fitted_pipeline_warns(self):
        """Re-fitting (and discarding) existing statistics should never be silent."""
        pipeline = self.make_pipeline()  # fitted=True by default
        model = self.build_model(preprocessor=pipeline)
        self.assertTrue(pipeline.is_fitted)

        with self.assertWarns(UserWarning):
            model.train(
                self.shifted_X,
                self.raw_y,
                epochs=1,
                batch_size=8,
                verbose=0,
                print_every=0,
                fit_preprocessor=True,
            )

    def test_fit_preprocessor_none_first_fit_does_not_warn(self):
        """Auto-fitting a never-fitted pipeline is the ordinary path -- silent."""
        pipeline = self.make_pipeline(fitted=False)
        model = self.build_model(preprocessor=pipeline)
        self.assertFalse(pipeline.is_fitted)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.train(
                self.raw_X, self.raw_y, epochs=1, batch_size=8, verbose=0, print_every=0
            )

        self.assertEqual(caught, [])

    def test_fit_preprocessor_false_never_fits(self):
        pipeline = self.make_pipeline()
        model = self.build_model(preprocessor=pipeline)
        before = float(pipeline.transforms[-1].mean.ravel()[0])

        model.train(
            self.shifted_X,
            self.raw_y,
            epochs=1,
            batch_size=8,
            verbose=0,
            print_every=0,
            fit_preprocessor=False,
        )
        self.assertAlmostEqual(
            float(pipeline.transforms[-1].mean.ravel()[0]), before, places=6
        )

    def test_fit_preprocessor_false_leaves_an_unfitted_pipeline_unfitted(self):
        pipeline = self.make_pipeline(fitted=False)
        model = self.build_model(preprocessor=pipeline)

        # train() itself now guards this case with a clear RuntimeError (matching
        # evaluate()/predict()) instead of quietly fitting it on the way past.
        with self.assertRaises(RuntimeError):
            model.train(
                self.raw_X,
                self.raw_y,
                epochs=1,
                batch_size=8,
                verbose=0,
                print_every=0,
                fit_preprocessor=False,
            )
        self.assertFalse(pipeline.is_fitted)

    # ---- Unfitted-pipeline guards ----

    def test_evaluate_raises_on_an_unfitted_pipeline(self):
        model = self.build_model(preprocessor=self.make_pipeline(fitted=False))
        with self.assertRaises(RuntimeError) as ctx:
            model.evaluate(self.raw_X, self.raw_y, verbose=0)
        self.assertIn("not fitted", str(ctx.exception))

    def test_predict_raises_on_an_unfitted_pipeline(self):
        model = self.build_model(preprocessor=self.make_pipeline(fitted=False))
        with self.assertRaises(RuntimeError) as ctx:
            model.predict(self.raw_X)
        self.assertIn("not fitted", str(ctx.exception))

    # ---- Hot loop ----

    def test_transform_is_applied_per_batch_during_train(self):
        spy = SpyCountingTransform()
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            spy,
        ])
        model = self.build_model(preprocessor=pipeline)
        model.train(
            self.raw_X,
            self.raw_y,
            epochs=1,
            batch_size=8,
            shuffle=False,
            verbose=0,
            print_every=0,
        )

        # One single-sample device probe, then one call per mini-batch.
        self.assertEqual(spy.batch_lengths, [1, 8, 8, 8, 8])

    def test_transform_is_applied_per_batch_during_evaluate(self):
        spy = SpyCountingTransform()
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            spy,
        ])
        model = self.build_model(preprocessor=pipeline)
        model.evaluate(self.raw_X, self.raw_y, batch_size=16, verbose=0)

        self.assertEqual(spy.batch_lengths, [1, 16, 16])

    def test_transform_is_applied_per_batch_during_predict(self):
        spy = SpyCountingTransform()
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            spy,
        ])
        model = self.build_model(preprocessor=pipeline)
        model.predict(self.raw_X, batch_size=16)

        # predict() now probes device like train()/evaluate() -- one single-sample
        # device probe, then one call per mini-batch.
        self.assertEqual(spy.batch_lengths, [1, 16, 16])

    # ---- predict() output placement ----

    def test_predict_streams_to_host_under_a_device_moving_pipeline(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        preds = model.predict(self.raw_X, batch_size=8, stream_to_host=True)

        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.shape, (self.NUM_SAMPLES, self.NUM_CLASSES))

    def test_predict_keeps_output_on_device_when_not_streaming(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        preds = model.predict(self.raw_X, batch_size=8, stream_to_host=False)

        self.assertIsInstance(preds, self.xp.ndarray)
        self.assertEqual(preds.shape, (self.NUM_SAMPLES, self.NUM_CLASSES))

    def test_predict_batch_size_invariance_under_a_pipeline(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        full = model.predict(self.raw_X)
        batched = model.predict(self.raw_X, batch_size=8)

        np.testing.assert_allclose(full, batched, rtol=1e-5, atol=1e-5)


base_case.register_test_suites(globals(), TestModelPreprocessorBase)
