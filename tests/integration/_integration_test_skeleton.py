"""
Aether-ML Integration Testing Template Skeleton
==============================================
This template serves as a standardized blueprint for integration tests that
orchestrate multiple components (Model, layers, losses, optimizers,
preprocessing) across NumPy (CPU) and CuPy (GPU) backends.

Architecture Pattern:
- Creates test classes per backend target using factory metaprogramming
  (`make_integration_suite`), same convention as tests/_skeleton.py and
  tests/optimizers/adam_base_suite.py.
- Registers generated classes into `globals()` so `python3 -m unittest discover`
  picks them up.


Notes for anyone extending this skeleton:

- For the fused `SoftmaxCategoricalCrossEntropy` loss, `loss.backward(...)`
  expects the *post-softmax* probabilities (`loss.output`), not raw model
  logits -- see how `tests/losses/test_softmax_categorical_crossentropy.py`
  calls it. If you're hand-rolling a training step instead of calling
  `model.train()`, use `model.loss.output`, not the raw `model.forward()`
  return value.
"""
import unittest

import aether as ae
import aether.config as config
from tests.base_case import BACKENDS_TO_TEST, AetherBaseTestCase


# Dynamic Factory Class Generation for Integration Tests
def make_integration_suite(backend_name):
    class_name = f"Test_Integration_{backend_name.upper()}"

    # Rewrite 'TestIntegration' for the specific components being tested
    class TestIntegration(AetherBaseTestCase):
        """Integration test suite executing multi-component flows."""
        SEED = 42
        NUM_CLASSES = 10

        def setUp(self):
            self.backend_name = backend_name
            config.set_backend(backend_name=self.backend_name)
            self.xp = config.xp

            # Build synthetic integration data
            self.xp.random.seed(self.SEED)
            self.batch_size = 8
            self.input_shape = (self.batch_size, 28, 28, 1)

            self.X_train = self.xp.random.randn(*self.input_shape).astype(self.xp.float32)
            self.y_train = self.xp.random.randint(0, self.NUM_CLASSES, size=(self.batch_size,))

            # Build your architecture here, e.g.:
            #
            # self.model = ae.Model()
            # self.manual_seed()
            # self.model.add(ae.Flatten())
            # self.model.add(ae.Dense(28 * 28 * 1, 32))
            # self.model.add(ae.ReLU())
            # self.model.add(ae.Dense(32, self.NUM_CLASSES))
            # self.model.configure(
            #     loss=ae.SoftmaxCategoricalCrossEntropy(),
            #     optimizer=ae.Adam(),
            #     accuracy=ae.CategoricalAccuracy(),
            # )
            # self.model.set_precision()
            # self.model.to(self.backend_name)   # BEFORE finalize()
            # self.model.finalize()

        def test_pipeline_execution(self):
            """Verify data flows through preprocessing and layers without errors."""
            pass

        def test_end_to_end_training_step(self):
            """Verify forward -> loss -> backward -> optimizer step reduces loss.

            Prefer driving this through Model.train() rather than reimplementing
            the loop by hand -- Model.train() is itself the composition under
            test, and a hand-rolled loop can silently paper over a wiring bug
            that only shows up through the public API.
            """
            pass

        def test_device_execution(self):
            """Verify computations execute natively on the target array backend."""
            self.assertIs(type(self.X_train), self.xp.ndarray)

    # Metaprogramming class property remapping for clear test-runner outputs
    TestIntegration.__name__ = class_name
    TestIntegration.__qualname__ = class_name

    return TestIntegration


# Global Registration for Test Discovery
for backend in BACKENDS_TO_TEST:
    suite_cls = make_integration_suite(backend_name=backend)
    globals()[suite_cls.__name__] = suite_cls