import contextlib
import io
import unittest

import numpy as np

import aether.config as config
from aether.layers import Dense, SoftMax
from aether.losses import CategoricalCrossEntropy
from aether.metrics import CategoricalAccuracy
from aether.model import Model
from tests.base_case import register_test_suites
from tests.model.base import ModelBaseTestCase


class TestModelEvaluateBase(ModelBaseTestCase):
    __test__ = False

    def setUp(self):
        super().setUp()
        self.model = Model()
        self.model.to(self.backend_name)
        self.model.add(Dense(n_inputs=4, n_neurons=self.NUM_CLASSES))
        self.model.add(SoftMax())

        self.loss = CategoricalCrossEntropy()
        self.accuracy = CategoricalAccuracy()
        self.model.configure(loss=self.loss, accuracy=self.accuracy)

    def test_evaluate_unfinalized_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            self.model.evaluate(self.X_val, self.y_val, verbose=False)

    def test_evaluate_full_batch(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        val_loss, val_acc = self.model.evaluate(
            self.X_val, self.y_val, batch_size=None, verbose=False
        )

        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        self.assertGreater(val_loss, 0.0)
        self.assertTrue(0.0 <= val_acc <= 1.0)

    def test_evaluate_mini_batch_matches_full_batch(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))

        full_loss, full_acc = self.model.evaluate(
            self.X_val, self.y_val, batch_size=None, verbose=False
        )

        batch_loss, batch_acc = self.model.evaluate(
            self.X_val, self.y_val, batch_size=4, verbose=False
        )

        self.assertAlmostEqual(full_loss, batch_loss, places=4)
        self.assertAlmostEqual(full_acc, batch_acc, places=4)

    def test_evaluate_without_loss(self):
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(n_inputs=4, n_neurons=self.NUM_CLASSES))
        model.add(SoftMax())
        model.configure(accuracy=self.accuracy)
        model.finalize(input_shape=(self.NUM_FEATURES,))

        val_loss, val_acc = model.evaluate(self.X_val, self.y_val, verbose=False)
        self.assertEqual(val_loss, 0.0)
        self.assertTrue(0.0 <= val_acc <= 1.0)

    def test_evaluate_does_not_mutate_parameters(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        trainable_layer = self.model.trainable_layers[0]
        initial_weights = self.xp.copy(trainable_layer.weights)

        self.model.evaluate(self.X_val, self.y_val, verbose=False)

        self.assertTrue(
            self.xp.array_equal(initial_weights, trainable_layer.weights),
            "Evaluation must not modify layer weights.",
        )

    def test_evaluate_device_mismatch_raises_type_error(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))
        if self.backend_name == "cupy":
            X_host = np.random.randn(len(self.X_val), self.NUM_FEATURES).astype("float32")
            y_host = np.random.randint(0, self.NUM_CLASSES, size=(len(self.X_val),)).astype("int32")
            with self.assertRaises(TypeError):
                self.model.evaluate(X_host, y_host, verbose=False)
        elif config.HAS_CUPY:
            import cupy as cp
            X_gpu = cp.asarray(self.X_val)
            y_gpu = cp.asarray(self.y_val)
            with self.assertRaises(TypeError):
                self.model.evaluate(X_gpu, y_gpu, verbose=False)

    def test_evaluate_verbose_output(self):
        self.model.finalize(input_shape=(self.NUM_FEATURES,))

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.model.evaluate(self.X_val, self.y_val, verbose=True)
        self.assertIn("[Validation]", buffer.getvalue())

        buffer_silent = io.StringIO()
        with contextlib.redirect_stdout(buffer_silent):
            self.model.evaluate(self.X_val, self.y_val, verbose=False)
        self.assertEqual(buffer_silent.getvalue(), "")


register_test_suites(globals(), TestModelEvaluateBase)