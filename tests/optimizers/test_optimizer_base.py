import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.layers.linear import Dense
from aether.optimizers.adam import Adam, Optimizer


class BaseTestOptimizerBase(base_case.AetherBaseLayerTestCase):
    __test__ = False

    def test_l2_gradient_matches_loss_derivative(self):
        """d/dw of Loss.regularization_loss's 0.5 * l2 * w**2 is l2 * w, not 2 * l2 * w --
        base Optimizer and Adam must agree on this for both weights and biases."""
        layer = self.make_built_layer(
            Dense,
            input_shape=(4,),
            seed=0,
            n_inputs=4,
            n_neurons=3,
            l2=(0.1, 0.1),
        )
        layer.dweights = self.xp.zeros_like(layer.weights)
        layer.dbiases = self.xp.zeros_like(layer.biases)

        expected_dweights = config.to_device(0.1 * layer.weights, target="numpy")
        expected_dbiases = config.to_device(0.1 * layer.biases, target="numpy")

        for optimizer in (Optimizer(), Adam()):
            dweights, dbiases = optimizer._get_regularized_gradients(layer, self.xp)
            np.testing.assert_allclose(
                config.to_device(dweights, target="numpy"), expected_dweights
            )
            np.testing.assert_allclose(
                config.to_device(dbiases, target="numpy"), expected_dbiases
            )

    def test_get_config_works_for_a_custom_subclass(self):
        """The base config must not read attributes that only Adam defines."""
        class SGD(Optimizer):
            def step(self):
                pass

        self.assertEqual(SGD(lr=0.1).get_config(), {"lr": 0.1, "decay": 0.0})
        self.assertIn("epsilon", Adam().get_config())


base_case.register_test_suites(globals(), BaseTestOptimizerBase)
