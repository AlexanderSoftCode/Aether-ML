import numpy as np
from tests.base_case import register_test_suites
from tests.model.base import ModelBaseTestCase
from aether.model import Model
from aether.layers.linear import Dense
from aether.losses import SoftmaxCategoricalCrossEntropy


class TestModelPredict(ModelBaseTestCase):
    __test__ = False

    def test_predict_logits_vs_probabilities(self):
        """Verify predict() routes output through fused activation or returns raw logits."""
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        model.configure(loss=SoftmaxCategoricalCrossEntropy())
        model.finalize((self.NUM_FEATURES,))

        # 1. Probabilities (SoftMax applied -> sum along axis=1 is 1.0)
        probs = model.predict(self.X_val, batch_size=8, return_logits=False)
        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape, (len(self.X_val), self.NUM_CLASSES))

        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums, dtype=np.float32), atol=1e-4)

        # 2. Raw logits (unfused dense activations -> sum along axis=1 is not 1.0)
        logits = model.predict(self.X_val, batch_size=8, return_logits=True)
        self.assertIsInstance(logits, np.ndarray)
        self.assertEqual(logits.shape, (len(self.X_val), self.NUM_CLASSES))
        self.assertFalse(np.allclose(np.sum(logits, axis=1), 1.0, atol=1e-2))

    def test_predict_unfinalized_raises_runtime_error(self):
        """Verify predict() raises RuntimeError if called prior to finalize()."""
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        model.configure(loss=SoftmaxCategoricalCrossEntropy())

        with self.assertRaises(RuntimeError):
            model.predict(self.X_val)

    def test_predict_rejects_invalid_batch_size_and_empty_input(self):
        """Verify a non-positive batch_size or empty X raises ValueError instead of
        dividing by zero or returning None."""
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        model.finalize((self.NUM_FEATURES,))

        for batch_size in (0, -1):
            with self.subTest(batch_size=batch_size), self.assertRaisesRegex(ValueError, "batch_size"):
                model.predict(self.X_val, batch_size=batch_size)

        with self.assertRaisesRegex(ValueError, "empty input"):
            model.predict(self.X_val[:0])

    def test_predict_stream_to_host_flag(self):
        """Verify stream_to_host controls whether outputs are NumPy or device arrays."""
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        model.configure(loss=SoftmaxCategoricalCrossEntropy())
        model.finalize((self.NUM_FEATURES,))

        # stream_to_host=True -> Always NumPy
        out_host = model.predict(self.X_val, stream_to_host=True)
        self.assertIsInstance(out_host, np.ndarray)

        # stream_to_host=False -> Matches active array module (xp)
        out_device = model.predict(self.X_val, stream_to_host=False)
        expected_type = self.xp.ndarray
        self.assertIsInstance(out_device, expected_type)

    def test_predict_batch_size_invariance(self):
        """Verify predict produces identical numerical results regardless of batch_size."""
        model = Model()
        model.to(self.backend_name)
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        model.configure(loss=SoftmaxCategoricalCrossEntropy())
        model.finalize((self.NUM_FEATURES,))

        full_batch_preds = model.predict(self.X_val, batch_size=None)
        mini_batch_preds = model.predict(self.X_val, batch_size=7)  # Uneven batch division

        np.testing.assert_allclose(full_batch_preds, mini_batch_preds, atol=1e-6)

    def test_inference_cache_invalidation(self):
        """Verify inference passes do not retain training backward caches in Dense layers."""
        model = Model()
        model.to(self.backend_name)
        dense = Dense(self.NUM_FEATURES, self.NUM_CLASSES)
        model.add(dense)
        model.configure(loss=SoftmaxCategoricalCrossEntropy())
        model.finalize((self.NUM_FEATURES,))

        _ = model.predict(self.X_val)

        self.assertIsNone(dense._inputs_compute)
        self.assertIsNone(dense._weights_compute)


register_test_suites(globals(), TestModelPredict)