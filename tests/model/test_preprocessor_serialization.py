import json
import os
import shutil
import tempfile
import warnings
import zipfile

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
    Rescale,
    StandardScaler,
    ToTensor,
)


class TestPreprocessorSerializationBase(base_case.AetherBaseTestCase):
    __test__ = False

    NUM_SAMPLES = 24
    NUM_FEATURES = 6
    NUM_CLASSES = 3

    def setUp(self):
        super().setUp()
        if not base_case.HAS_SAFETENSORS:
            self.skipTest("safetensors library is not installed in the environment.")

        np.random.seed(5)
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, "model.aether")
        self.resave_path = os.path.join(self.temp_dir, "model_resaved.aether")

        self.raw_X = np.random.randint(
            0, 256, size=(self.NUM_SAMPLES, self.NUM_FEATURES), dtype=np.uint8
        )
        self.raw_y = np.random.randint(
            0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)
        ).astype(np.int32)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        super().tearDown()

    # ---- Helpers ----

    def make_pipeline(self, nested=False, scaler=None):
        scaler = StandardScaler(axis=0) if scaler is None else scaler
        to_tensor = ToTensor(dtype="float32", target_device=self.backend_name)
        rescale = Rescale(factor=1.0 / 255.0)
        members = (
            [to_tensor, Compose([rescale, scaler])]
            if nested
            else [to_tensor, rescale, scaler]
        )
        return Compose(members).fit(self.raw_X)

    def build_model(self, preprocessor=None, device=None, finalize=True):
        model = Model()
        model.manual_seed(9)
        model.add(Dense(self.NUM_FEATURES, 8))
        model.add(ReLU())
        model.add(Dense(8, self.NUM_CLASSES))
        model.configure(
            loss=CategoricalCrossEntropy(),
            optimizer=Adam(lr=0.01),
            accuracy=CategoricalAccuracy(),
            preprocessor=preprocessor,
        )
        model.to(device or self.backend_name)
        if finalize:
            model.finalize((self.NUM_FEATURES,))
        return model

    def read_manifest(self, path):
        with zipfile.ZipFile(path, "r") as zipf:
            return json.loads(zipf.read("architecture.json").decode("utf-8"))

    def read_manifest_bytes(self, path):
        with zipfile.ZipFile(path, "r") as zipf:
            return zipf.read("architecture.json")

    def rewrite_manifest(self, src, dst, mutate):
        """Rebuilds an archive with a mutated manifest, weights untouched."""
        with zipfile.ZipFile(src, "r") as zipf:
            manifest = json.loads(zipf.read("architecture.json").decode("utf-8"))
            weights_bytes = zipf.read("weights.safetensors")

        mutate(manifest)

        with zipfile.ZipFile(dst, "w") as zipf:
            zipf.writestr(
                "architecture.json",
                json.dumps(manifest, indent=2).encode("utf-8"),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            zipf.writestr(
                "weights.safetensors", weights_bytes, compress_type=zipfile.ZIP_STORED
            )
        return dst

    # ---- Round-trip stability ----

    def test_resave_produces_a_byte_identical_manifest(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        model.save(self.save_path)

        Model.load(self.save_path).save(self.resave_path)

        self.assertEqual(
            self.read_manifest_bytes(self.save_path),
            self.read_manifest_bytes(self.resave_path),
        )

    def test_resave_is_byte_identical_under_a_float16_policy(self):
        model = self.build_model(preprocessor=self.make_pipeline(), finalize=False)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*NumPy float16 is emulated.*", category=UserWarning
            )
            model.set_precision("float16")
            model.finalize((self.NUM_FEATURES,))
            model.save(self.save_path)
            Model.load(self.save_path).save(self.resave_path)

        self.assertEqual(
            self.read_manifest_bytes(self.save_path),
            self.read_manifest_bytes(self.resave_path),
        )

    def test_evaluate_parity_across_save_and_load_on_raw_input(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        model.train(
            self.raw_X, self.raw_y, epochs=2, batch_size=8, verbose=0, print_every=0
        )
        model.save(self.save_path)

        before = model.evaluate(self.raw_X, self.raw_y, batch_size=8, verbose=0)
        after = Model.load(self.save_path).evaluate(
            self.raw_X, self.raw_y, batch_size=8, verbose=0
        )

        self.assertAlmostEqual(before[0], after[0], places=5)
        self.assertAlmostEqual(before[1], after[1], places=5)

    def test_loaded_model_consumes_raw_input_without_reconfiguring(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        model.save(self.save_path)

        loaded = Model.load(self.save_path)
        preds = loaded.predict(self.raw_X, batch_size=8)

        self.assertEqual(preds.shape, (self.NUM_SAMPLES, self.NUM_CLASSES))
        self.assertTrue(bool(np.all(np.isfinite(preds))))

    def test_nested_compose_round_trips(self):
        model = self.build_model(preprocessor=self.make_pipeline(nested=True))
        model.save(self.save_path)

        restored = Model.load(self.save_path).preprocessor
        self.assertIsInstance(restored.transforms[1], Compose)
        self.assertIsInstance(restored.transforms[1].transforms[1], StandardScaler)
        self.assertTrue(restored.is_fitted)

    # ---- Null pipeline ----

    def test_null_preprocessor_never_appears_in_the_manifest(self):
        model = self.build_model()
        self.assertIsInstance(model.preprocessor, NullPreprocessor)
        model.save(self.save_path)

        self.assertIsNone(self.read_manifest(self.save_path)["preprocessor"])

    def test_null_preprocessor_is_reinstalled_on_load(self):
        self.build_model().save(self.save_path)
        self.assertIsInstance(Model.load(self.save_path).preprocessor, NullPreprocessor)

    def test_manifest_without_a_preprocessor_key_still_loads(self):
        """Schema '1.0' archives predate the key entirely."""
        self.build_model().save(self.save_path)

        legacy_path = os.path.join(self.temp_dir, "legacy.aether")

        def strip(manifest):
            manifest.pop("preprocessor", None)
            manifest["schema_version"] = "1.0"

        self.rewrite_manifest(self.save_path, legacy_path, strip)

        loaded = Model.load(legacy_path)
        self.assertIsInstance(loaded.preprocessor, NullPreprocessor)
        self.assertFalse(loaded._has_pipeline())

    # ---- dtype pin ----

    def test_pinned_scaler_dtype_survives_a_round_trip(self):
        model = self.build_model(
            preprocessor=self.make_pipeline(scaler=StandardScaler(axis=0, dtype="float32"))
        )
        model.save(self.save_path)

        restored = Model.load(self.save_path).preprocessor.transforms[-1]
        self.assertTrue(restored._dtype_pinned)
        self.assertEqual(restored.dtype, np.dtype("float32"))

    def test_inferred_scaler_dtype_comes_back_unpinned(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        model.save(self.save_path)

        restored = Model.load(self.save_path).preprocessor.transforms[-1]
        self.assertFalse(restored._dtype_pinned)
        self.assertEqual(restored.dtype, np.dtype("float32"))

    def test_restored_statistics_are_float32_not_float64(self):
        model = self.build_model(preprocessor=self.make_pipeline())
        model.save(self.save_path)

        restored = Model.load(self.save_path).preprocessor.transforms[-1]
        self.assertEqual(restored.mean.dtype, np.dtype("float32"))
        self.assertEqual(restored.std.dtype, np.dtype("float32"))

    # ---- Cross-device load ----

    def test_saved_on_cupy_loads_onto_the_host(self):
        if self.backend_name != "cupy":
            self.skipTest("Only meaningful when the archive was written from CuPy.")

        model = self.build_model(preprocessor=self.make_pipeline(), device="cupy")
        model.save(self.save_path)

        with warnings.catch_warnings():
            # The restored ToTensor keeps its saved "cupy" pin, which mismatches
            # device="numpy" here -- not what this test is checking, so it's
            # silenced rather than left to leak into the test run.
            warnings.simplefilter("ignore", category=UserWarning)
            loaded = Model.load(self.save_path, device="numpy")
        scaler = loaded.preprocessor.transforms[-1]

        self.assertEqual(loaded.device, "numpy")
        self.assertIsInstance(loaded.layers[0].weights, np.ndarray)
        self.assertIsInstance(scaler.mean, np.ndarray)
        self.assertEqual(loaded.preprocessor.transforms[0].target_device, "numpy")
        self.assertIsInstance(loaded.predict(self.raw_X, batch_size=8), np.ndarray)

    # ---- Schema version gate ----

    def test_higher_minor_schema_warns_and_loads(self):
        self.build_model(preprocessor=self.make_pipeline()).save(self.save_path)
        newer_path = os.path.join(self.temp_dir, "newer_minor.aether")
        self.rewrite_manifest(
            self.save_path,
            newer_path,
            lambda manifest: manifest.update({"schema_version": "1.99"}),
        )

        with self.assertWarns(UserWarning):
            loaded = Model.load(newer_path)
        self.assertTrue(loaded.is_finalized)
        self.assertTrue(loaded._has_pipeline())

    def test_higher_major_schema_raises(self):
        self.build_model(preprocessor=self.make_pipeline()).save(self.save_path)
        newer_path = os.path.join(self.temp_dir, "newer_major.aether")
        self.rewrite_manifest(
            self.save_path,
            newer_path,
            lambda manifest: manifest.update({"schema_version": "2.0"}),
        )

        with self.assertRaises(ValueError) as ctx:
            Model.load(newer_path)
        self.assertIn("2.0", str(ctx.exception))

    def test_current_schema_version_is_written(self):
        self.build_model(preprocessor=self.make_pipeline()).save(self.save_path)
        self.assertEqual(self.read_manifest(self.save_path)["schema_version"], "1.1")

    # ---- Error paths surfaced through Model.load ----

    def test_unknown_preprocessor_class_raises_value_error(self):
        self.build_model(preprocessor=self.make_pipeline()).save(self.save_path)
        corrupt_path = os.path.join(self.temp_dir, "corrupt_preprocessor.aether")

        def inject(manifest):
            manifest["preprocessor"] = {
                "class_name": "HypotheticalWhiteningTransform",
                "config": {},
            }

        self.rewrite_manifest(self.save_path, corrupt_path, inject)

        with self.assertRaises(ValueError) as ctx:
            Model.load(corrupt_path)
        self.assertIn("HypotheticalWhiteningTransform", str(ctx.exception))

    def test_base_preprocess_class_in_manifest_raises_type_error(self):
        self.build_model(preprocessor=self.make_pipeline()).save(self.save_path)
        corrupt_path = os.path.join(self.temp_dir, "corrupt_base.aether")

        self.rewrite_manifest(
            self.save_path,
            corrupt_path,
            lambda manifest: manifest.update(
                {"preprocessor": {"class_name": "Preprocess", "config": {}}}
            ),
        )

        with self.assertRaises(TypeError):
            Model.load(corrupt_path)


base_case.register_test_suites(globals(), TestPreprocessorSerializationBase)
