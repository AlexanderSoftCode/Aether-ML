import json
import os
import shutil
import struct
import tempfile
import warnings
import zipfile

import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.layers.activations import LeakyReLU, ReLU
from aether.layers.conv import Conv2d
from aether.layers.linear import Dense, Flatten
from aether.layers.normalization import BatchNorm
from aether.layers.pooling import MaxPool2d
from aether.losses import CategoricalCrossEntropy, SoftmaxCategoricalCrossEntropy
from aether.metrics import CategoricalAccuracy
from aether.model import Model
from aether.optimizers import Adam, AdamW


class TestModelLoadBase(base_case.AetherBaseTestCase):
    __test__ = False
    
    NUM_FEATURES = 8
    NUM_CLASSES = 3

    def setUp(self):
        super().setUp()
        if not base_case.HAS_SAFETENSORS:
            self.skipTest("safetensors library is not installed in the environment.")

        # Seed both NumPy and the active backend uniformly
        self.xp.random.seed(42)
        np.random.seed(42)

        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, "test_model.aether")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        super().tearDown()

    # ---- Error Handling & Malformed Archive Tests ----

    def test_load_nonexistent_file_raises(self):
        """Ensure attempting to load a nonexistent path raises FileNotFoundError or OSError."""
        non_existent_path = os.path.join(self.temp_dir, "ghost_model.aether")
        with self.assertRaises((FileNotFoundError, OSError)):
            Model.load(non_existent_path)

    def test_load_corrupted_archive_raises(self):
        """Ensure loading a corrupted/truncated zip archive raises ValueError."""
        corrupt_path = os.path.join(self.temp_dir, "corrupt_model.aether")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_VALID_ZIP_PAYLOAD_DATA")

        with self.assertRaisesRegex(ValueError, "not a valid .aether archive"):
            Model.load(corrupt_path)

    def test_load_unknown_layer_class_raises(self):
        """Ensure manifest referencing an unresolvable layer class raises ValueError."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 16))
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            manifest = json.loads(zipf.read("architecture.json").decode("utf-8"))
            weights_bytes = zipf.read("weights.safetensors")

        manifest["layers"][0]["class_name"] = "HypotheticalAlienLayer"
        corrupt_manifest_bytes = json.dumps(manifest).encode("utf-8")

        corrupt_save_path = os.path.join(self.temp_dir, "corrupt_layer.aether")
        with zipfile.ZipFile(corrupt_save_path, "w") as zipf:
            zipf.writestr("architecture.json", corrupt_manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
            zipf.writestr("weights.safetensors", weights_bytes, compress_type=zipfile.ZIP_STORED)

        with self.assertRaises(ValueError) as ctx:
            Model.load(corrupt_save_path)
        self.assertIn("HypotheticalAlienLayer", str(ctx.exception))

    def _save_mlp_members(self):
        """Saves a small MLP and returns its (manifest bytes, weights bytes)."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 16))
        model.add(ReLU())
        model.add(Dense(16, self.NUM_CLASSES))
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            return zipf.read("architecture.json"), zipf.read("weights.safetensors")

    def _write_archive(self, *members):
        """Writes (name, data[, compress_type]) members verbatim to a fresh archive."""
        path = os.path.join(self.temp_dir, "crafted.aether")
        with warnings.catch_warnings(), zipfile.ZipFile(path, "w") as zipf:
            warnings.simplefilter("ignore")  # duplicate-name warning
            for name, data, *compress_type in members:
                zipf.writestr(name, data, compress_type=compress_type[0] if compress_type else None)
        return path

    def _write_manifest_archive(self, mutate):
        """Re-saves the MLP with its manifest dict passed through `mutate`."""
        manifest_bytes, weights_bytes = self._save_mlp_members()
        manifest = json.loads(manifest_bytes)
        mutate(manifest)
        return self._write_archive(
            ("architecture.json", json.dumps(manifest).encode()), ("weights.safetensors", weights_bytes)
        )

    @staticmethod
    def _patch_central_directory(path, member, offset, fmt, value):
        """Overwrites one field of `member`'s central-directory record in place."""
        with open(path, "r+b") as f:
            blob = f.read()
            pos = blob.index(b"PK\x01\x02")
            while blob[pos + 46:pos + 46 + len(member)] != member.encode():
                pos = blob.index(b"PK\x01\x02", pos + 4)
            f.seek(pos + offset)
            f.write(struct.pack(fmt, value))

    def _save_with_rewritten_weights(self, rewrite):
        """Saves a small MLP, then rewrites its weight dict via `rewrite`."""
        import safetensors.numpy

        manifest_bytes, weights_bytes = self._save_mlp_members()
        weights = safetensors.numpy.load(weights_bytes)
        rewrite(weights)
        return self._write_archive(
            ("architecture.json", manifest_bytes),
            ("weights.safetensors", safetensors.numpy.save(weights)),
        )

    def test_load_missing_weights_raises(self):
        """Ensure an archive lacking a parameter fails instead of keeping random init."""
        bad_path = self._save_with_rewritten_weights(lambda w: w.pop("2.weights"))
        with self.assertRaisesRegex(ValueError, "2.weights"):
            Model.load(bad_path)

    def test_load_unexpected_weights_raises(self):
        """Ensure an archive carrying a parameter the graph has no slot for fails."""
        def add_stray(weights):
            weights["7.weights"] = np.zeros(1, dtype=np.float32)

        bad_path = self._save_with_rewritten_weights(add_stray)
        with self.assertRaisesRegex(ValueError, "7.weights"):
            Model.load(bad_path)

    def test_load_mis_shaped_weights_raises(self):
        """Ensure a parameter whose shape disagrees with the rebuilt layer fails."""
        def reshape(weights):
            weights["0.weights"] = np.zeros((2, 2), dtype=np.float32)

        bad_path = self._save_with_rewritten_weights(reshape)
        with self.assertRaisesRegex(ValueError, "0.weights"):
            Model.load(bad_path)

    def test_load_wrong_dtype_weights_raises(self):
        """Ensure a correctly shaped parameter of the wrong dtype fails, since the
        fused GPU kernels would otherwise write float32 past the end of its buffer."""
        def downcast(weights):
            weights["0.weights"] = weights["0.weights"].astype(np.uint8)

        bad_path = self._save_with_rewritten_weights(downcast)
        with self.assertRaisesRegex(ValueError, "0.weights.*uint8"):
            Model.load(bad_path)

    # ---- Untrusted Archive Hardening Tests ----

    def test_load_rejects_zip_bomb_manifest_without_inflating(self):
        """A highly compressible manifest is rejected on its declared size."""
        _, weights_bytes = self._save_mlp_members()
        bomb = self._write_archive(
            ("architecture.json", b" " * (2 << 20), zipfile.ZIP_DEFLATED),
            ("weights.safetensors", weights_bytes),
        )
        self.assertLess(os.path.getsize(bomb), 64 << 10)
        with self.assertRaisesRegex(ValueError, "max_manifest_bytes"):
            Model.load(bomb)

    def test_load_rejects_member_with_lying_size_header(self):
        """A bomb whose central directory understates its size is still read bounded."""
        _, weights_bytes = self._save_mlp_members()
        bomb = self._write_archive(
            ("architecture.json", b" " * (2 << 20), zipfile.ZIP_DEFLATED),
            ("weights.safetensors", weights_bytes),
        )
        self._patch_central_directory(bomb, "architecture.json", 24, "<I", 10)  # file_size
        with self.assertRaises(ValueError):
            Model.load(bomb)

    def test_load_weights_over_limit_names_kwarg(self):
        """Weights larger than max_weights_bytes are rejected with the kwarg to raise."""
        self._save_mlp_members()
        with self.assertRaisesRegex(ValueError, "max_weights_bytes=100;"):
            Model.load(self.save_path, max_weights_bytes=100)

    def test_load_rejects_file_larger_than_combined_cap(self):
        """The on-disk size is capped before the central directory is parsed."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 4096))
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)
        with self.assertRaisesRegex(ValueError, "on disk"):
            Model.load(self.save_path, max_manifest_bytes=1000, max_weights_bytes=1000)

    def test_load_rejects_fifo_without_blocking(self):
        """A path that is not a regular file is rejected instead of hanging the open."""
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFOs are POSIX-only")
        fifo = os.path.join(self.temp_dir, "pipe.aether")
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            Model.load(fifo)

    def test_load_rejects_unexpected_archive_members(self):
        """Archives must hold exactly the two members, stored as save() writes them."""
        manifest_bytes, weights_bytes = self._save_mlp_members()
        cases = {
            "extra member": [("architecture.json", manifest_bytes), ("weights.safetensors", weights_bytes),
                             ("payload.bin", b"x")],
            "duplicate manifest": [("architecture.json", manifest_bytes), ("architecture.json", b"{}"),
                                   ("weights.safetensors", weights_bytes)],
            "missing weights": [("architecture.json", manifest_bytes)],
            "deflated weights": [("architecture.json", manifest_bytes),
                                 ("weights.safetensors", weights_bytes, zipfile.ZIP_DEFLATED)],
        }
        for case, members in cases.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                Model.load(self._write_archive(*members))

    def test_load_rejects_encrypted_member(self):
        """A member flagged as encrypted is rejected instead of raising RuntimeError."""
        self._save_mlp_members()
        self._patch_central_directory(self.save_path, "architecture.json", 8, "<H", 0x1)  # flag_bits
        with self.assertRaisesRegex(ValueError, "encrypted"):
            Model.load(self.save_path)

    def test_load_rejects_malformed_manifest(self):
        """Structurally invalid manifests raise ValueError, never another exception type."""
        def set_layer(key, value):
            return lambda m: m["layers"][0].__setitem__(key, value)

        mutations = {
            "input_shape str": lambda m: m.__setitem__("input_shape", "abc"),
            "input_shape negative": lambda m: m.__setitem__("input_shape", [-1]),
            "input_shape bool": lambda m: m.__setitem__("input_shape", [True]),
            "input_shape missing": lambda m: m.pop("input_shape"),
            "class_name int": set_layer("class_name", 3),
            "config str": set_layer("config", "abc"),
            "index mismatch": set_layer("index", 5),
            "NaN": lambda m: m.__setitem__("seed", float("nan")),
            "compile entry str": lambda m: m.__setitem__("compile", {"loss": "Loss"}),
        }
        for case, mutate in mutations.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                Model.load(self._write_manifest_archive(mutate))

        manifest_bytes, weights_bytes = self._save_mlp_members()
        raw_manifests = {
            "top level list": b"[]",
            "duplicate keys": manifest_bytes.rstrip()[:-1] + b', "layers": []}',
            "deep nesting": b"[" * 10_000 + b"]" * 10_000,
            "invalid utf-8": b"\xff\xfe{}",
        }
        for case, raw in raw_manifests.items():
            path = self._write_archive(("architecture.json", raw), ("weights.safetensors", weights_bytes))
            with self.subTest(case=case), self.assertRaises(ValueError):
                Model.load(path)

    def test_load_rejects_garbage_weights(self):
        """Weights that are not safetensors raise ValueError."""
        manifest_bytes, _ = self._save_mlp_members()
        path = self._write_archive(("architecture.json", manifest_bytes), ("weights.safetensors", b"\x08" * 64))
        with self.assertRaisesRegex(ValueError, "safetensors"):
            Model.load(path)

    def test_load_rejects_oversized_dimensions_before_allocating(self):
        """A tiny archive cannot request a huge allocation through its manifest."""
        def huge_dense(manifest):
            manifest["layers"][0]["config"].update(n_inputs=10**7, n_neurons=10**7)

        with self.assertRaisesRegex(ValueError, "dimension"):
            Model.load(self._write_manifest_archive(huge_dense))

    def test_failed_load_leaves_backend_unchanged(self):
        """A rejected archive never switches the process-wide backend."""
        if not config.HAS_CUPY:
            self.skipTest("needs a second backend")
        other = "numpy" if self.backend_name == "cupy" else "cupy"
        bad_path = self._save_with_rewritten_weights(lambda w: w.pop("2.weights"))
        with self.assertRaises(ValueError):
            Model.load(bad_path, device=other)
        self.assertIs(config.xp, self.xp)

    # ---- Basic Architecture & Parameter Hydration Tests ----

    def test_basic_mlp_graph_reconstruction_and_parameter_parity(self):
        """Verify reconstructed MLP matches original layer classes, configs, and exact tensor weights."""
        orig_model = Model()
        orig_model.manual_seed(123)
        d1 = Dense(self.NUM_FEATURES, 16, l2=(1e-4))
        act = ReLU()
        d2 = Dense(16, self.NUM_CLASSES)

        orig_model.add(d1)
        orig_model.add(act)
        orig_model.add(d2)
        orig_model.finalize((self.NUM_FEATURES,))
        orig_model.save(self.save_path)

        loaded_model = Model.load(self.save_path)

        self.assertTrue(loaded_model.is_finalized)
        self.assertEqual(len(loaded_model.layers), 3)
        self.assertEqual(type(loaded_model.layers[0]), Dense)
        self.assertEqual(type(loaded_model.layers[1]), ReLU)
        self.assertEqual(type(loaded_model.layers[2]), Dense)

        self.assertEqual(loaded_model.layers[0].n_inputs, self.NUM_FEATURES)
        self.assertEqual(loaded_model.layers[0].n_neurons, 16)
        self.assertEqual(loaded_model.layers[0].weight_regularizer_l2, 1e-4)

        np.testing.assert_allclose(
            config.to_device(loaded_model.layers[0].weights, target="numpy"),
            config.to_device(d1.weights, target="numpy"),
        )
        np.testing.assert_allclose(
            config.to_device(loaded_model.layers[0].biases, target="numpy"),
            config.to_device(d1.biases, target="numpy"),
        )
        np.testing.assert_allclose(
            config.to_device(loaded_model.layers[2].weights, target="numpy"),
            config.to_device(d2.weights, target="numpy"),
        )
        np.testing.assert_allclose(
            config.to_device(loaded_model.layers[2].biases, target="numpy"),
            config.to_device(d2.biases, target="numpy"),
        )

    # ---- Inference & Prediction Parity Tests ----

    def test_forward_pass_and_prediction_numerical_parity(self):
        """Verify predictions of loaded model are identical to the source model in evaluation mode."""
        orig_model = Model()
        orig_model.manual_seed(99)
        orig_model.add(Dense(self.NUM_FEATURES, 32))
        orig_model.add(LeakyReLU(alpha=0.2))
        orig_model.add(Dense(32, self.NUM_CLASSES))
        orig_model.finalize((self.NUM_FEATURES,))
        orig_model.save(self.save_path)

        loaded_model = Model.load(self.save_path)

        raw_X = np.random.randn(10, self.NUM_FEATURES).astype(np.float32)
        X = config.to_device(raw_X, target=self.backend_name)

        orig_preds = orig_model.predict(X)
        loaded_preds = loaded_model.predict(X)

        np.testing.assert_allclose(
            config.to_device(loaded_preds, target="numpy"),
            config.to_device(orig_preds, target="numpy"),
            rtol=1e-5,
            atol=1e-6,
        )

    # ---- Precision Policy Restoration Tests ----

    def test_precision_policy_restoration(self):
        """Assert target compute dtype policy persists through serialization and re-hydrates properly."""
        orig_model = Model()
        orig_model.manual_seed(42)
        orig_model.add(Dense(self.NUM_FEATURES, 16))
        orig_model.add(ReLU())
        orig_model.add(Dense(16, self.NUM_CLASSES))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            orig_model.set_precision("float32")

        orig_model.finalize((self.NUM_FEATURES,))
        orig_model.save(self.save_path)

        loaded_model = Model.load(self.save_path)

        self.assertIsNotNone(loaded_model.precision_policy)
        self.assertEqual(loaded_model.precision_policy.compute_dtype_name, "float32")
        self.assertEqual(loaded_model.layers[0].weights.dtype, np.float32)
        self.assertEqual(loaded_model.layers[2].weights.dtype, np.float32)

    # ---- Complex CNN Architecture & Running Buffers ----

    def test_complex_cnn_architecture_and_batchnorm_buffers_load(self):
        """Test loading of a full CNN graph and verify persistent buffers (BatchNorm running stats)."""
        orig_model = Model()
        orig_model.manual_seed(77)

        conv = Conv2d(
            in_channels=3,
            out_channels=8,
            filter_size=(3, 3),
            stride=(1, 1),
            padding="same",
        )
        bn = BatchNorm()
        pool = MaxPool2d(filter_size=(2, 2), stride=(2, 2))
        flatten = Flatten()
        dense = Dense(4 * 4 * 8, self.NUM_CLASSES)

        orig_model.add(conv)
        orig_model.add(bn)
        orig_model.add(ReLU())
        orig_model.add(pool)
        orig_model.add(flatten)
        orig_model.add(dense)
        orig_model.finalize((8, 8, 3))

        raw_input = np.random.randn(4, 8, 8, 3).astype(np.float32)
        X_sample = config.to_device(raw_input, target=self.backend_name)
        orig_model.forward(X_sample, training=True)

        orig_model.save(self.save_path)
        loaded_model = Model.load(self.save_path)

        self.assertEqual(len(loaded_model.layers), 6)
        self.assertIsInstance(loaded_model.layers[0], Conv2d)
        self.assertIsInstance(loaded_model.layers[1], BatchNorm)
        self.assertIsInstance(loaded_model.layers[3], MaxPool2d)
        self.assertIsInstance(loaded_model.layers[4], Flatten)
        self.assertIsInstance(loaded_model.layers[5], Dense)

        np.testing.assert_allclose(
            config.to_device(loaded_model.layers[0].weights, target="numpy"),
            config.to_device(conv.weights, target="numpy"),
        )
        np.testing.assert_allclose(
            config.to_device(loaded_model.layers[0].biases, target="numpy"),
            config.to_device(conv.biases, target="numpy"),
        )

        loaded_bn = loaded_model.layers[1]
        np.testing.assert_allclose(
            config.to_device(loaded_bn.gamma, target="numpy"),
            config.to_device(bn.gamma, target="numpy"),
        )
        np.testing.assert_allclose(
            config.to_device(loaded_bn.beta, target="numpy"),
            config.to_device(bn.beta, target="numpy"),
        )
        np.testing.assert_allclose(
            config.to_device(loaded_bn.running_mean, target="numpy"),
            config.to_device(bn.running_mean, target="numpy"),
        )
        np.testing.assert_allclose(
            config.to_device(loaded_bn.running_var, target="numpy"),
            config.to_device(bn.running_var, target="numpy"),
        )

        orig_eval = orig_model.forward(X_sample, training=False)
        loaded_eval = loaded_model.forward(X_sample, training=False)

        np.testing.assert_allclose(
            config.to_device(loaded_eval, target="numpy"),
            config.to_device(orig_eval, target="numpy"),
            rtol=1e-3,
            atol=1e-3,
        )

    # ---- Explicit Target Device Migration Tests ----

    def test_explicit_device_migration_on_load(self):
        """Verify Model.load(..., device=target) places parameters onto requested backend."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 4))
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        loaded_model = Model.load(self.save_path, device="numpy")
        self.assertEqual(loaded_model.device, "numpy")
        self.assertIsInstance(loaded_model.layers[0].weights, np.ndarray)

    # ---- Training Component & Compilation Deserialization Tests ----

    def test_load_reconstructs_full_compilation_components(self):
        """Verify loss, optimizer, and accuracy re-hydrate with exact hyperparameters."""
        orig_model = Model()
        orig_model.manual_seed(42)
        orig_model.add(Dense(self.NUM_FEATURES, 16))
        orig_model.add(ReLU())
        orig_model.add(Dense(16, self.NUM_CLASSES))

        loss = CategoricalCrossEntropy(label_smoothing=0.02)
        optimizer = Adam(lr=2e-3, decay=1e-4, beta_1=0.92, beta_2=0.99)
        accuracy = CategoricalAccuracy()

        orig_model.configure(loss=loss, optimizer=optimizer, accuracy=accuracy)
        orig_model.finalize((self.NUM_FEATURES,))
        orig_model.save(self.save_path)

        loaded_model = Model.load(self.save_path)

        # Assert loss restored
        self.assertIsInstance(loaded_model.loss, CategoricalCrossEntropy)
        self.assertEqual(loaded_model.loss.label_smoothing, 0.02)

        # Assert optimizer restored
        self.assertIsInstance(loaded_model.optimizer, Adam)
        self.assertEqual(loaded_model.optimizer.lr, 2e-3)
        self.assertEqual(loaded_model.optimizer.decay, 1e-4)
        self.assertEqual(loaded_model.optimizer.beta_1, 0.92)
        self.assertEqual(loaded_model.optimizer.beta_2, 0.99)

        # Assert accuracy restored
        self.assertIsInstance(loaded_model.accuracy, CategoricalAccuracy)

    def test_load_reconstructs_fused_loss_and_adamw(self):
        """Verify SoftmaxCategoricalCrossEntropy fused loss and AdamW are restored."""
        orig_model = Model()
        orig_model.manual_seed(42)
        orig_model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))

        loss = SoftmaxCategoricalCrossEntropy(label_smoothing=0.05)
        optimizer = AdamW(lr=1e-3, weight_decay=1e-2)

        orig_model.configure(loss=loss, optimizer=optimizer)
        orig_model.finalize((self.NUM_FEATURES,))
        orig_model.save(self.save_path)

        loaded_model = Model.load(self.save_path)

        self.assertIsInstance(loaded_model.loss, SoftmaxCategoricalCrossEntropy)
        self.assertEqual(loaded_model.loss.label_smoothing, 0.05)
        self.assertIsInstance(loaded_model.optimizer, AdamW)
        self.assertEqual(loaded_model.optimizer.lr, 1e-3)
        self.assertEqual(loaded_model.optimizer.weight_decay, 1e-2)

    def test_load_unknown_component_class_raises_value_error(self):
        """Ensure manifest referencing an unresolvable loss/optimizer raises ValueError."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 16))
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            manifest = json.loads(zipf.read("architecture.json").decode("utf-8"))
            weights_bytes = zipf.read("weights.safetensors")

        # Inject an alien optimizer
        manifest["compile"]["optimizer"] = {
            "class_name": "HypotheticalQuantumOptimizer",
            "config": {"lr": 0.01},
        }

        corrupt_manifest_bytes = json.dumps(manifest).encode("utf-8")
        corrupt_save_path = os.path.join(self.temp_dir, "corrupt_opt.aether")
        with zipfile.ZipFile(corrupt_save_path, "w") as zipf:
            zipf.writestr("architecture.json", corrupt_manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
            zipf.writestr("weights.safetensors", weights_bytes, compress_type=zipfile.ZIP_STORED)

        with self.assertRaises(ValueError) as ctx:
            Model.load(corrupt_save_path)
        self.assertIn("HypotheticalQuantumOptimizer", str(ctx.exception))

    def test_load_invalid_component_subclass_raises_type_error(self):
        """Ensure manifest pointing to a non-subclass component raises TypeError."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 16))
        model.finalize((self.NUM_FEATURES,))
        model.save(self.save_path)

        with zipfile.ZipFile(self.save_path, "r") as zipf:
            manifest = json.loads(zipf.read("architecture.json").decode("utf-8"))
            weights_bytes = zipf.read("weights.safetensors")

        # Point loss to a class that exists in the namespace but is not a Loss subclass
        manifest["compile"]["loss"] = {
            "class_name": "Loss",  # Base class cannot be instantiated as a valid loss directly
            "config": {},
        }

        corrupt_manifest_bytes = json.dumps(manifest).encode("utf-8")
        corrupt_save_path = os.path.join(self.temp_dir, "corrupt_subclass.aether")
        with zipfile.ZipFile(corrupt_save_path, "w") as zipf:
            zipf.writestr("architecture.json", corrupt_manifest_bytes, compress_type=zipfile.ZIP_DEFLATED)
            zipf.writestr("weights.safetensors", weights_bytes, compress_type=zipfile.ZIP_STORED)

        with self.assertRaises((TypeError, ValueError)):
            Model.load(corrupt_save_path)

    def test_loaded_model_evaluation_and_loss_parity(self):
        """Verify model.evaluate() works out-of-the-box post-load and matches original output exactly."""
        orig_model = Model()
        orig_model.manual_seed(42)
        orig_model.add(Dense(self.NUM_FEATURES, 16))
        orig_model.add(ReLU())
        orig_model.add(Dense(16, self.NUM_CLASSES))

        orig_model.configure(
            loss=CategoricalCrossEntropy(),
            optimizer=Adam(lr=1e-3),
            accuracy=CategoricalAccuracy(),
        )
        # Migrate original model to active test backend (CuPy or NumPy)
        orig_model.to(self.backend_name)
        orig_model.finalize((self.NUM_FEATURES,))
        orig_model.save(self.save_path)

        # Loaded model automatically adopts the active backend
        loaded_model = Model.load(self.save_path)

        raw_X = np.random.randn(20, self.NUM_FEATURES).astype(np.float32)
        raw_y = np.random.randint(0, self.NUM_CLASSES, size=(20,)).astype(np.int32)

        X = config.to_device(raw_X, target=self.backend_name)
        y = config.to_device(raw_y, target=self.backend_name)

        orig_loss, orig_acc = orig_model.evaluate(X=X, y=y, batch_size=10, verbose=0)
        load_loss, load_acc = loaded_model.evaluate(X=X, y=y, batch_size=10, verbose=0)

        # Confirm non-zero and exact equality
        self.assertGreater(load_loss, 0.0)
        self.assertAlmostEqual(orig_loss, load_loss, places=5)
        self.assertAlmostEqual(orig_acc, load_acc, places=5)

    # ---- regression tests ------------------

    def test_save_and_load_without_explicit_extension(self):
        """Verify model can be saved and loaded without explicitly providing .aether suffix."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 16))
        model.finalize((self.NUM_FEATURES,))

        # Target path omitting .aether extension
        target_path_no_ext = os.path.join(self.temp_dir, "model_no_ext")
        expected_saved_file = target_path_no_ext + ".aether"

        # 1. Save without extension
        model.save(target_path_no_ext)
        self.assertTrue(
            os.path.isfile(expected_saved_file),
            f"Expected archive '{expected_saved_file}' to be created on disk.",
        )

        # 2. Load without extension
        loaded_model_no_ext = Model.load(target_path_no_ext)
        self.assertTrue(loaded_model_no_ext.is_finalized)
        self.assertEqual(len(loaded_model_no_ext.layers), 1)

        # 3. Load with explicit extension
        loaded_model_with_ext = Model.load(expected_saved_file)
        self.assertTrue(loaded_model_with_ext.is_finalized)
        self.assertEqual(len(loaded_model_with_ext.layers), 1)
base_case.register_test_suites(globals(), TestModelLoadBase)