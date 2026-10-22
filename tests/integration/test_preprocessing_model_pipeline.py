import numpy as np

import aether as ae
import aether.config as config
from aether.preprocessing import Compose, Rescale, StandardScaler, ToTensor
from tests.base_case import register_test_suites
from tests.integration.model_base_suite import ModelIntegrationBaseCase


class TestPreprocessingModelPipeline(ModelIntegrationBaseCase):
    N_SAMPLES = 16

    def setUp(self):
        super().setUp()
        np.random.seed(self.SEED)
        self.raw_X = np.random.randint(
            0, 256, size=(self.N_SAMPLES, 32, 32, 3), dtype=np.uint8
        )
        self.raw_y = np.random.randint(
            0, self.NUM_CLASSES, size=(self.N_SAMPLES,)
        ).astype(np.int64)

    def test_compose_pipeline_feeds_cnn_model_forward(self):
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
        ])
        X = pipeline(self.raw_X)

        model = self.build_cnn_model()
        out = model.forward(X, training=False)

        self.assertEqual(out.shape, (self.N_SAMPLES, self.NUM_CLASSES))
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(out))))

    def test_standard_scaler_fit_transform_feeds_mlp_training(self):
        X_float = (
            config.to_device(self.raw_X, target=self.backend_name).astype(
                self.xp.float32
            )
            / 255.0
        )
        scaler = StandardScaler(axis=(0, 1, 2)).fit(X_float)
        X_scaled = scaler(X_float)
        y = config.to_device(self.raw_y, target=self.backend_name)

        model = self.build_mlp_model()
        model.train(
            X=X_scaled,
            y=y,
            epochs=1,
            batch_size=8,
            verbose=False,
            print_every=10**9,
        )

        preds = model.predict(X_scaled, batch_size=8)
        self.assertEqual(preds.shape, (self.N_SAMPLES, self.NUM_CLASSES))
        # predict() streams to NumPy by default, so check via numpy
        self.assertTrue(bool(np.all(np.isfinite(preds))))

    def test_to_tensor_dtype_and_device_survive_into_model_forward(self):
        X = ae.to_tensor(
            self.raw_X, target_device=self.backend_name, dtype="float32"
        )

        self.assertIsInstance(X, self.xp.ndarray)
        self.assertEqual(str(X.dtype), "float32")

        model = self.build_mlp_model()
        out = model.forward(X, training=False)

        self.assertEqual(str(out.dtype), "float32")
        self.assertIsInstance(out, self.xp.ndarray)

    def test_rescale_only_pipeline_bounds_conv_input_range(self):
        """Regression guard: forgetting Rescale (leaving raw 0-255 uint8-cast
        values flowing into Conv) is a realistic mistake -- verify a
        correctly-scaled pipeline actually produces values in [0, 1] before
        they hit the model, rather than trusting Conv to tolerate either."""
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
        ])
        X = pipeline(self.raw_X)
        self.assertGreaterEqual(float(self.xp.min(X)), 0.0)
        self.assertLessEqual(float(self.xp.max(X)), 1.0)


register_test_suites(globals(), TestPreprocessingModelPipeline)