import json
import os
import shutil
import tempfile
import warnings
import zipfile
from unittest import mock
import numpy as np

import aether.config as config
import tests.base_case as base_case

from aether.layers.activations import LeakyReLU, ReLU
from aether.layers.conv import Conv2d
from aether.layers.linear import Dense, Flatten
from aether.layers.normalization import BatchNorm
from aether.layers.pooling import MaxPool2d
from aether.losses import CategoricalCrossEntropy, SoftmaxCategoricalCrossEntropy
from aether.optimizers import Adam, AdamW
from aether.metrics import Accuracy
from aether.model import Model



class TestModelSaveBase(base_case.AetherBaseTestCase):
    __test__ = False

    NUM_FEATURES = 8
    NUM_CLASSES = 3

    def setUp(self):
        super().setUp()
        if not base_case.HAS_SAFETENSORS:
            self.skipTest("safetensors library is not installed in the environment.")

        self.xp.random.seed(42)
        np.random.seed(42)

        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, "test_model.aether")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        super().tearDown()

    # ---- Precondition & Guardrail Tests ----

    def test_save_unfinalized_model_raises(self):
        """Ensure calling save() on an unfinalized graph immediately raises RuntimeError."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 4))

        with self.assertRaises(RuntimeError):
            model.save(self.save_path)

    def test_failed_save_keeps_existing_archive(self):
        """A save that dies mid-write leaves the previous archive intact and no temp files."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 4))
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)
        with open(self.save_path, "rb") as f:
            original = f.read()

        with (
            mock.patch("aether.model.os.fsync", side_effect=OSError("No space left on device")),
            self.assertRaises(OSError),
        ):
            model.save(self.save_path)

        with open(self.save_path, "rb") as f:
            self.assertEqual(f.read(), original)
        self.assertEqual(os.listdir(self.temp_dir), ["test_model.aether"])

    def test_save_permissions_follow_umask(self):
        """The atomic temp file must not leave archives at tempfile's private 0600."""
        if os.name == "nt":
            self.skipTest("POSIX permission bits")
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 4))
        model.finalize((self.NUM_FEATURES,))
        for umask, expected in ((0o022, 0o644), (0o002, 0o664), (0o077, 0o600)):
            previous = os.umask(umask)
            try:
                model.save(self.save_path)
            finally:
                os.umask(previous)
            with self.subTest(umask=oct(umask)):
                self.assertEqual(os.stat(self.save_path).st_mode & 0o777, expected)

    def test_save_replaces_symlink_instead_of_following_it(self):
        """Saving over a symlink swaps the link, leaving its target untouched."""
        target = os.path.join(self.temp_dir, "elsewhere.txt")
        with open(target, "wb") as f:
            f.write(b"keep")
        os.symlink(target, self.save_path)

        model = Model()
        model.add(Dense(self.NUM_FEATURES, 4))
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        self.assertFalse(os.path.islink(self.save_path))
        with open(target, "rb") as f:
            self.assertEqual(f.read(), b"keep")
        self.assertTrue(zipfile.is_zipfile(self.save_path))

    def test_save_nonfinite_config_raises(self):
        """NaN/Infinity cannot be written as strict JSON, so save() names the value."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 4))
        model.add(LeakyReLU(alpha=float("nan")))
        model.finalize((self.NUM_FEATURES,))
        with self.assertRaisesRegex(ValueError, r"layers\.1\.config\.alpha"):
            model.save(self.save_path)
        self.assertFalse(os.path.exists(self.save_path))

    # ---- Archive Container & Compression Tests ----

    def test_archive_structure_and_compression_modes(self):
        """Verify .aether zip bundle contains both manifest and weights with correct compression flags."""
        model = Model()
        model.manual_seed(100)
        model.add(Dense(self.NUM_FEATURES, 16))
        model.add(ReLU())
        model.add(Dense(16, self.NUM_CLASSES))
        model.finalize((self.NUM_FEATURES,))

        model.save(self.save_path)
        self.assertTrue(os.path.exists(self.save_path))

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            namelist = zipf.namelist()
            self.assertIn("architecture.json", namelist)
            self.assertIn("weights.safetensors", namelist)

            # Ensure JSON is compressed and weights are stored uncompressed for zero-copy memory mapping
            info_json = zipf.getinfo("architecture.json")
            info_weights = zipf.getinfo("weights.safetensors")
            self.assertEqual(info_json.compress_type, zipfile.ZIP_DEFLATED)
            self.assertEqual(info_weights.compress_type, zipfile.ZIP_STORED)

    # ---- Manifest Integrity & Layer Metadata Tests ----

    def test_manifest_metadata_and_layer_configs(self):
        """Assert schema_version, input_shape, precision policy, and layer configs serialize accurately."""
        model = Model()
        model.manual_seed(42)
        model.add(
            Dense(
                self.NUM_FEATURES,
                16,
                l2=(1e-4, 1e-4),
            )
        )
        model.add(LeakyReLU(alpha=0.15))
        model.add(Dense(16, self.NUM_CLASSES))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.set_precision("float32")

        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            manifest = json.loads(
                zipf.read("architecture.json").decode("utf-8")
            )

        self.assertEqual(manifest["schema_version"], "1.1")
        self.assertEqual(manifest["input_shape"], [self.NUM_FEATURES])
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["precision_policy"], "float32")
        self.assertEqual(len(manifest["layers"]), 3)

        # Validate Dense config round-trip
        dense_cfg = manifest["layers"][0]
        self.assertEqual(dense_cfg["class_name"], "Dense")
        self.assertEqual(
            dense_cfg["config"]["n_inputs"], self.NUM_FEATURES
        )
        self.assertEqual(dense_cfg["config"]["n_neurons"], 16)
        self.assertEqual(dense_cfg["config"]["l2"][0], 1e-4)

        # Validate LeakyReLU config
        leaky_cfg = manifest["layers"][1]
        self.assertEqual(leaky_cfg["class_name"], "LeakyReLU")
        self.assertEqual(leaky_cfg["config"]["alpha"], 0.15)

        # Validate parameter-free fallback
        out_dense_cfg = manifest["layers"][2]
        self.assertEqual(out_dense_cfg["class_name"], "Dense")
        self.assertEqual(out_dense_cfg["config"]["n_inputs"], 16)
        self.assertEqual(
            out_dense_cfg["config"]["n_neurons"], self.NUM_CLASSES
        )

    # ---- Parameter Extraction, Contiguity, and Cross-Backend Tests ----

    def test_safetensors_tensor_payload_and_contiguity(self):
        """Verify tensors convert to CPU NumPy, remain C-contiguous, and match memory parameter values."""
        model = Model()
        dense1 = Dense(self.NUM_FEATURES, 16)
        dense2 = Dense(16, self.NUM_CLASSES)
        model.add(dense1)
        model.add(ReLU())
        model.add(dense2)
        model.finalize((self.NUM_FEATURES,))

        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            weights_bytes = zipf.read("weights.safetensors")
            loaded_tensors = base_case.safetensors_np.load(weights_bytes)

        expected_keys = {"0.weights", "0.biases", "2.weights", "2.biases"}
        self.assertEqual(set(loaded_tensors.keys()), expected_keys)

        for key, tensor in loaded_tensors.items():
            self.assertIsInstance(tensor, np.ndarray)
            self.assertTrue(
                tensor.flags.c_contiguous,
                f"Tensor '{key}' is not C-contiguous.",
            )

        # Compare against live layer state
        np.testing.assert_allclose(
            loaded_tensors["0.weights"],
            config.to_device(dense1.weights, target="numpy"),
        )
        np.testing.assert_allclose(
            loaded_tensors["0.biases"],
            config.to_device(dense1.biases, target="numpy"),
        )
        np.testing.assert_allclose(
            loaded_tensors["2.weights"],
            config.to_device(dense2.weights, target="numpy"),
        )
        np.testing.assert_allclose(
            loaded_tensors["2.biases"],
            config.to_device(dense2.biases, target="numpy"),
        )

    # ---- Complex Multi-Layer Graph (Conv, Pool, BatchNorm) ----

    def test_complex_cnn_architecture_save(self):
        """Test serialization of a full CNN pipeline containing Conv, MaxPool2d, BatchNorm, and Flatten."""
        model = Model()
        conv = Conv2d(
            in_channels=3,
            out_channels=8,
            filter_size=(3, 3),
            stride=(1, 1),
            padding="same",
        )
        bn = BatchNorm()
        pool = MaxPool2d(filter_size=(2, 2), stride=(2, 2))
        dense = Dense(4 * 4 * 8, self.NUM_CLASSES)

        model.add(conv)
        model.add(bn)
        model.add(ReLU())
        model.add(pool)
        model.add(Flatten())
        model.add(dense)
        model.finalize((8, 8, 3))

        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            manifest = json.loads(
                zipf.read("architecture.json").decode("utf-8")
            )
            weights = base_case.safetensors_np.load(
                zipf.read("weights.safetensors")
            )

        # Verify Conv configuration and parameters
        self.assertEqual(manifest["layers"][0]["class_name"], "Conv2d")
        self.assertEqual(
            manifest["layers"][0]["config"]["filter_size"], [3, 3]
        )
        self.assertIn("0.weights", weights)
        self.assertIn("0.biases", weights)
        self.assertEqual(weights["0.weights"].shape, (3, 3, 3, 8))

        # Verify BatchNorm buffers & parameters
        self.assertEqual(manifest["layers"][1]["class_name"], "BatchNorm")
        self.assertIn("1.gamma", weights)
        self.assertIn("1.beta", weights)
        self.assertIn("1.running_mean", weights)
        self.assertIn("1.running_var", weights)

        # Verify Flatten has empty config
        self.assertEqual(manifest["layers"][4]["class_name"], "Flatten")
        self.assertEqual(manifest["layers"][4]["config"], {})

        # Verify Dense parameters
        self.assertIn("5.weights", weights)
        self.assertIn("5.biases", weights)

        # ---- Compile Dictionary Serialization Tests ----

    def test_save_unconfigured_model_serializes_null_compile_entries(self):
        """Verify unconfigured/uncompiled models serialize null/None entries in the compile section."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 16))
        model.finalize((self.NUM_FEATURES,))

        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            manifest = json.loads(zipf.read("architecture.json").decode("utf-8"))

        self.assertIn("compile", manifest)
        compile_cfg = manifest["compile"]
        self.assertIsNone(compile_cfg["loss"])
        self.assertIsNone(compile_cfg["optimizer"])
        self.assertIsNone(compile_cfg["accuracy"])

    def test_save_configured_training_components_manifest(self):
        """Verify compiled loss, optimizer, and accuracy configs serialize accurately to compile key."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 16))
        model.add(ReLU())
        model.add(Dense(16, self.NUM_CLASSES))

        loss = CategoricalCrossEntropy(label_smoothing=0.01)
        optimizer = Adam(lr=1e-3, decay=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        accuracy = Accuracy()

        model.configure(
            loss=loss,
            optimizer=optimizer,
            accuracy=accuracy,
        )
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            manifest = json.loads(zipf.read("architecture.json").decode("utf-8"))

        self.assertIn("compile", manifest)
        compile_cfg = manifest["compile"]

        # Validate Loss serialization
        self.assertIsNotNone(compile_cfg["loss"])
        self.assertEqual(compile_cfg["loss"]["class_name"], "CategoricalCrossEntropy")
        self.assertEqual(compile_cfg["loss"]["config"]["label_smoothing"], 0.01)

        # Validate Optimizer serialization
        self.assertIsNotNone(compile_cfg["optimizer"])
        self.assertEqual(compile_cfg["optimizer"]["class_name"], "Adam")
        opt_cfg = compile_cfg["optimizer"]["config"]
        self.assertEqual(opt_cfg["lr"], 1e-3)
        self.assertEqual(opt_cfg["decay"], 5e-5)
        self.assertEqual(opt_cfg["beta_1"], 0.9)
        self.assertEqual(opt_cfg["beta_2"], 0.999)
        self.assertEqual(opt_cfg["epsilon"], 1e-7)

        # Validate Accuracy serialization
        self.assertIsNotNone(compile_cfg["accuracy"])
        self.assertEqual(compile_cfg["accuracy"]["class_name"], "Accuracy")
        self.assertIsInstance(compile_cfg["accuracy"]["config"], dict)

    def test_save_partial_configuration_with_fused_loss_and_adamw(self):
        """Verify partial compilation with fused SoftmaxCategoricalCrossEntropy and AdamW serializes correctly."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 16))

        loss = SoftmaxCategoricalCrossEntropy(label_smoothing=0.05)
        optimizer = AdamW(lr=1e-3, weight_decay=1e-2, beta_1=0.9, beta_2=0.999)

        model.configure(
            loss=loss,
            optimizer=optimizer,
        )
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            manifest = json.loads(zipf.read("architecture.json").decode("utf-8"))

        self.assertIn("compile", manifest)
        compile_cfg = manifest["compile"]

        # Validate Fused Loss Serialization
        self.assertIsNotNone(compile_cfg["loss"])
        self.assertEqual(compile_cfg["loss"]["class_name"], "SoftmaxCategoricalCrossEntropy")
        self.assertEqual(compile_cfg["loss"]["config"]["label_smoothing"], 0.05)

        # Validate AdamW Optimizer Serialization
        self.assertIsNotNone(compile_cfg["optimizer"])
        self.assertEqual(compile_cfg["optimizer"]["class_name"], "AdamW")
        opt_cfg = compile_cfg["optimizer"]["config"]
        self.assertEqual(opt_cfg["lr"], 1e-3)
        self.assertEqual(opt_cfg["weight_decay"], 1e-2)

        # Accuracy omitted
        self.assertIsNone(compile_cfg["accuracy"])
base_case.register_test_suites(globals(), TestModelSaveBase)