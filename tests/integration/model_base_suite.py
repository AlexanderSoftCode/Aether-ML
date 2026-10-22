"""
Shared fixture-building helpers for Model-level integration tests.
"""
import math

import aether as ae
import aether.config as config
import tests.base_case as base_case


class ModelIntegrationBaseCase(base_case.AetherBaseTestCase):
    __test__ = False

    NUM_CLASSES = 10
    SEED = 42

    def make_synthetic_image_data(self, n_samples=16, height=32, width=32, channels=3):
        self.xp.random.seed(self.SEED)
        X = self.xp.random.randn(n_samples, height, width, channels).astype(self.xp.float32)
        y = self.xp.random.randint(0, self.NUM_CLASSES, size=(n_samples,)).astype(self.xp.int64)
        return X, y

    def build_cnn_model(self, *, device=None, precision=None, input_dim=(32, 32, 3)):
        target_device = device if device is not None else self.backend_name
        model = ae.Model()
        model.add(ae.Conv2d(3, 32, (3, 3), (1, 1), padding="same"))
        model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
        model.add(ae.ReLU())
        model.add(ae.Conv2d(32, 64, (3, 3), (1, 1), padding="same"))
        model.add(ae.AvgPool2d((2, 2), (2, 2), padding="valid"))
        model.add(ae.LeakyReLU(alpha=0.01))
        model.add(ae.SpatialDropout(rate=0.05, seed=self.SEED))
        model.add(ae.GlobalAvgPool())
        model.add(ae.Dense(64, self.NUM_CLASSES))
        model.configure(
            loss=ae.SoftmaxCategoricalCrossEntropy(label_smoothing=0.01),
            optimizer=ae.Adam(lr=0.001, decay=5e-5),
            accuracy=ae.CategoricalAccuracy(),
        )
        model.to(device if device is not None else self.backend_name)
        model.manual_seed(seed=42)
        
        return self._place_and_finalize(model, precision=precision, input_dim=input_dim)

    def build_mlp_model(self, *, device=None, precision=None, input_dim=(32, 32, 3)):
        target_device = device if device is not None else self.backend_name
        model = ae.Model()
        model.add(ae.Flatten())
        model.add(ae.Dense(math.prod(input_dim), 128))
        model.add(ae.ReLU())
        model.add(ae.Dense(128, self.NUM_CLASSES))
        model.configure(
            loss=ae.SoftmaxCategoricalCrossEntropy(label_smoothing=0.01),
            optimizer=ae.Adam(lr=0.001, decay=5e-5),
            accuracy=ae.CategoricalAccuracy(),
        )
        model.to(device if device is not None else self.backend_name)
        model.manual_seed(seed=42)
        return self._place_and_finalize(model, precision=precision, input_dim=input_dim)

    @staticmethod
    def _place_and_finalize(model, *, precision, input_dim):
        if precision is not None:
            model.set_precision(compute_dtype=precision)
        model.finalize(input_shape=input_dim)
        return model