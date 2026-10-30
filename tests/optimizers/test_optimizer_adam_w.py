import tests.base_case as base_case
from aether.optimizers.adam import AdamW
from tests.optimizers.adam_base_suite import BaseTestOptimizerAdam


class TestAdamW(BaseTestOptimizerAdam):
    OPTIMIZER_CLASS = AdamW

    def test_weight_decay_decoupled_update(self):
        """Verify decoupled weight decay behavior when gradients are zero."""
        self.layer.dweights = self.xp.zeros_like(self.layer.weights)
        self.layer.dbiases = self.xp.zeros_like(self.layer.biases)

        self.optimizer.weight_decay = 0.1
        lr = self.optimizer.current_lr
        weights_before = self.layer.weights.copy()
        biases_before = self.layer.biases.copy()

        self.optimizer.step()

        # With zero gradients, parameters should change strictly due to decoupled decay
        expected_weights = weights_before - (lr * 0.1 * weights_before)
        self.xp.testing.assert_array_almost_equal(
            self.layer.weights, expected_weights, decimal=5
        )

        # Biases must never be subject to weight decay
        self.xp.testing.assert_array_equal(self.layer.biases, biases_before)

    def test_no_weight_decay_flag_bypasses_decay(self):
        """Verify layer-level no_weight_decay flag bypasses decoupled weight decay."""
        self.layer.no_weight_decay = True
        self.layer.dweights = self.xp.zeros_like(self.layer.weights)
        self.layer.dbiases = self.xp.zeros_like(self.layer.biases)
        self.optimizer.weight_decay = 0.1
        weights_before = self.layer.weights.copy()

        self.optimizer.step()

        self.xp.testing.assert_array_equal(self.layer.weights, weights_before)


base_case.register_test_suites(globals(), TestAdamW)