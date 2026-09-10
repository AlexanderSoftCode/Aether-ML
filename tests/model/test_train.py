import io
import contextlib
import unittest
import numpy as np

import aether.config as config
from aether.model import Model
from aether.layers import Dense, ReLU
from aether.losses import CategoricalCrossEntropy
from aether.optimizers import Adam, AdamW
from aether.metrics import CategoricalAccuracy

from tests.base_case import register_test_suites
from tests.model.base import ModelBaseTestCase


class TestModelTrainBase(ModelBaseTestCase):
    __test__ = False

    def setUp(self):
        super().setUp()
        self.model = self._make_model()
        self.optimizer = self.model.optimizer
        self.accuracy = self.model.accuracy

    def test_train_unfinalized_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self.model.train(self.X, self.y, verbose=False)

    def test_train_without_loss_raises_runtime_error(self):
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(n_inputs=4, n_neurons=self.NUM_CLASSES))
        model.configure(optimizer=self.optimizer, accuracy=self.accuracy)
        model.finalize(input_shape=(self.NUM_FEATURES,))

        with self.assertRaises(RuntimeError):
            model.train(self.X, self.y, verbose=False)

    def test_train_updates_layer_parameters(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        trainable_layer = self.model.trainable_layers[0]

        initial_weights = self.xp.copy(trainable_layer.weights)
        self.model.train(self.X, self.y, epochs=1, batch_size=None, verbose=False)

        self.assertFalse(
            self.xp.allclose(initial_weights, trainable_layer.weights),
            "Expected weights to update after a training step."
        )

    def test_train_mini_batch_multiple_epochs(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        trainable_layer = self.model.trainable_layers[0]
        initial_weights = self.xp.copy(trainable_layer.weights)

        self.model.train(
            self.X,
            self.y,
            epochs=3,
            batch_size=8,
            shuffle=True,
            verbose=False
        )

        self.assertFalse(self.xp.allclose(initial_weights, trainable_layer.weights))

    def test_train_without_shuffle(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        trainable_layer = self.model.trainable_layers[0]
        initial_weights = self.xp.copy(trainable_layer.weights)

        self.model.train(
            self.X,
            self.y,
            epochs=2,
            batch_size=16,
            shuffle=False,
            verbose=False
        )

        self.assertFalse(self.xp.allclose(initial_weights, trainable_layer.weights))

    def test_train_with_regularization(self):
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(n_inputs=4, n_neurons=self.NUM_CLASSES, l2=(0.01)))
        model.configure(
            loss=CategoricalCrossEntropy(),
            optimizer=AdamW(lr=0.01),
            accuracy=CategoricalAccuracy()
        )
        model.finalize(input_shape=(self.NUM_FEATURES,))

        model.train(self.X, self.y, epochs=1, batch_size=16, verbose=False)

    def test_train_with_validation_data(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        self.model.train(
            self.X,
            self.y,
            epochs=1,
            batch_size=16,
            validation_data=(self.X_val, self.y_val),
            verbose=False
        )

    def test_train_device_mismatch_raises_type_error(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        if self.backend_name == "cupy":
            X_host = np.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
            y_host = np.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")
            with self.assertRaises(TypeError):
                self.model.train(X_host, y_host, verbose=False)
        elif config.HAS_CUPY:
            import cupy as cp
            X_gpu = cp.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
            y_gpu = cp.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")
            with self.assertRaises(TypeError):
                self.model.train(X_gpu, y_gpu, verbose=False)

    def test_train_verbose_output_suppression(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.model.train(self.X, self.y, epochs=1, verbose=False)
        self.assertEqual(buffer.getvalue(), "")

    def _make_model(self):
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(n_inputs=4, n_neurons=8))
        model.add(ReLU())
        model.add(Dense(n_inputs=8, n_neurons=self.NUM_CLASSES))
        model.configure(
            loss=CategoricalCrossEntropy(),
            optimizer=Adam(lr=0.01),
            accuracy=CategoricalAccuracy(),
        )
        return model

    def test_seeded_shuffle_is_reproducible(self):
        model_a = self._make_model()
        model_b = self._make_model()

        model_a.manual_seed(123)
        model_b.manual_seed(123)

        model_a.finalize(input_shape=(self.NUM_FEATURES,))
        model_b.finalize(input_shape=(self.NUM_FEATURES,))

        # batch_size=8 over NUM_SAMPLES=32 gives 4 batches/epoch, so shuffling
        # actually reorders which samples land in which batch.
        model_a.train(self.X, self.y, epochs=2, batch_size=8, shuffle=True, verbose=0)
        model_b.train(self.X, self.y, epochs=2, batch_size=8, shuffle=True, verbose=0)

        for layer_a, layer_b in zip(model_a.trainable_layers, model_b.trainable_layers):
            weights_a = config.to_device(layer_a.weights, target="numpy")
            weights_b = config.to_device(layer_b.weights, target="numpy")
            biases_a = config.to_device(layer_a.biases, target="numpy")
            biases_b = config.to_device(layer_b.biases, target="numpy")

            if self.backend_name == "cupy":
                np.testing.assert_allclose(weights_a, weights_b, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(biases_a, biases_b, rtol=1e-5, atol=1e-6)
            else:
                np.testing.assert_array_equal(weights_a, weights_b)
                np.testing.assert_array_equal(biases_a, biases_b)

    def test_telemetry_sync_suppressed_at_verbose_zero(self):
        # float() on the per-step metric is a GPU->host sync; verbose=0 must skip it.
        float_calls = []

        class _Synced:
            def __init__(self, value):
                self.value = value

            def __float__(self):
                float_calls.append(1)
                return float(self.value)

        class SpyAccuracy(CategoricalAccuracy):
            def calculate(self, predictions, y):
                return _Synced(super().calculate(predictions, y))

        for verbose, expect_sync in ((0, False), (2, True)):
            with self.subTest(verbose=verbose):
                float_calls.clear()
                model = self._make_model()
                model.configure(accuracy=SpyAccuracy())
                model.finalize(input_shape=(self.NUM_FEATURES,))
                with contextlib.redirect_stdout(io.StringIO()):
                    model.train(self.X, self.y, epochs=1, batch_size=8, verbose=verbose, print_every=1)
                self.assertEqual(bool(float_calls), expect_sync)

register_test_suites(globals(), TestModelTrainBase)