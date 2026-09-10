import aether as ae
import aether.config as config
from tests.base_case import register_test_suites
from tests.integration.model_base_suite import ModelIntegrationBaseCase


class TestBackendIsolation(ModelIntegrationBaseCase):
    """Guards against runtime forward/backward/step paths reading the mutable
    global config.xp instead of resolving the array module from the layer's
    own arrays -- a second model migrated/loaded onto the other device in
    the same process must not corrupt an already-finalized model."""

    BATCH_SIZE = 8

    def _build_model(self, device):
        model = ae.Model()
        model.add(ae.Conv2d(3, 8, (3, 3), (1, 1), padding="same"))
        model.add(ae.BatchNorm())
        model.add(ae.ReLU())
        model.add(ae.GlobalAvgPool())
        model.add(ae.Dense(8, self.NUM_CLASSES))
        model.configure(
            loss=ae.SoftmaxCategoricalCrossEntropy(),
            optimizer=ae.Adam(lr=0.001),
            accuracy=ae.CategoricalAccuracy(),
        )
        model.to(device)
        model.manual_seed(seed=42)
        model.finalize(input_shape=(32, 32, 3))
        return model

    def test_gpu_model_survives_global_backend_flip(self):
        if self.backend_name != "cupy":
            self.skipTest("Backend-isolation only meaningfully diverges on the CuPy backend.")

        model = self._build_model(device="cupy")
        X, y = self.make_synthetic_image_data()
        model.train(X, y, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)

        # Simulate a second model being loaded onto numpy in this process.
        config.set_backend("numpy")

        model.evaluate(X, y, verbose=0)
        model.predict(X)
        model.train(X, y, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)

    def test_cpu_model_survives_global_backend_flip(self):
        if self.backend_name != "cupy":
            self.skipTest("Backend-isolation only meaningfully diverges on the CuPy backend.")

        model = self._build_model(device="numpy")
        X, y = self.make_synthetic_image_data()
        X = config.to_device(X, target="numpy")
        y = config.to_device(y, target="numpy")

        # Simulate a second model being loaded onto cupy in this process.
        config.set_backend("cupy")

        model.train(X, y, epochs=1, batch_size=self.BATCH_SIZE, verbose=0)


register_test_suites(globals(), TestBackendIsolation)
