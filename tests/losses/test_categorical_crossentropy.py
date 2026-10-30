import tests.base_case as base_case
from aether.layers.linear import Dense
from aether.losses.categorical_crossentropy import CategoricalCrossEntropy


class TestLossCCE(base_case.AetherBaseTestCase):
    def setUp(self):
        super().setUp()
        self.layer = self.make_component(
            CategoricalCrossEntropy,
        )

    # ---- Forward ---------------------

    def test_forward_known_values(self):
        """Verify loss against a hand-calculated example (sparse labels)."""
        y_pred = self.xp.array([
            [0.7, 0.1, 0.2],
            [0.1, 0.5, 0.4],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 1])

        losses = self.layer.forward(y_pred, y_true, training=True)

        expected = -self.xp.log(self.xp.array([0.7, 0.5], dtype=self.xp.float32))
        self.xp.testing.assert_array_almost_equal(losses, expected, decimal=5)

    def test_forward_one_hot_matches_sparse(self):
        """Sparse and one-hot labels should produce identical losses."""
        y_pred = self.xp.array([
            [0.2, 0.7, 0.1],
            [0.6, 0.3, 0.1],
        ], dtype=self.xp.float32)
        y_true_sparse = self.xp.array([1, 0])
        y_true_onehot = self.xp.array([
            [0, 1, 0],
            [1, 0, 0],
        ], dtype=self.xp.float32)

        loss_sparse = self.layer.forward(y_pred, y_true_sparse, training=True)
        loss_onehot = self.layer.forward(y_pred, y_true_onehot, training=True)

        self.xp.testing.assert_array_almost_equal(loss_sparse, loss_onehot, decimal=5)

    def test_forward_clipping_prevents_log_zero(self):
        """Predictions of exactly 0 or 1 should not produce inf/nan in the loss."""
        y_pred = self.xp.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 2])

        losses = self.layer.forward(y_pred, y_true, training=True)

        self.assertFalse(self.xp.isnan(losses).any())
        self.assertFalse(self.xp.isinf(losses).any())

    def test_forward_label_smoothing_increases_loss_on_confident_prediction(self):
        """Label smoothing should raise the loss for an otherwise near-zero-loss prediction."""
        smoothed = self.make_component(
            CategoricalCrossEntropy, 
            label_smoothing=0.1,
        )

        y_pred = self.xp.array([[0.98, 0.01, 0.01]], dtype=self.xp.float32)
        y_true = self.xp.array([0])

        plain_loss = self.layer.forward(y_pred, y_true, training=True)
        smoothed_loss = smoothed.forward(y_pred, y_true, training=True)

        self.assertGreater(float(smoothed_loss[0]), float(plain_loss[0]))

    def test_forward_label_smoothing_ignored_when_not_training(self):
        """Label smoothing must not apply during eval/inference."""
        smoothed = self.make_component(
            CategoricalCrossEntropy, 
            label_smoothing=0.1,
        )

        y_pred = self.xp.array([[0.98, 0.01, 0.01]], dtype=self.xp.float32)
        y_true = self.xp.array([0])

        eval_loss = smoothed.forward(y_pred, y_true, training=False)
        plain_loss = self.layer.forward(y_pred, y_true, training=True)

        self.xp.testing.assert_array_almost_equal(eval_loss, plain_loss, decimal=5)

    # ---- Backward ---------------------

    def test_backward_gradient_shape(self):
        y_pred = self.xp.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 2])

        self.layer.backward(y_pred, y_true)

        self.assertEqual(self.layer.dinputs.shape, y_pred.shape)

    def test_backward_matches_formula(self):
        """dinputs should equal -y_true / clip(dvalues) / samples exactly."""
        y_pred = self.xp.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 1])

        self.layer.backward(y_pred, y_true)

        samples = y_pred.shape[0]
        n_classes = y_pred.shape[1]
        y_true_onehot = self.xp.eye(n_classes, dtype=self.xp.float32)[y_true]
        clipped = self.xp.clip(y_pred, 1e-7, 1 - 1e-7)
        expected = -y_true_onehot / clipped / samples

        self.xp.testing.assert_array_almost_equal(self.layer.dinputs, expected, decimal=5)

    def test_backward_numerical_gradient_check(self):
        """Finite-difference check: dinputs should equal the true gradient of forward()."""
        epsilon = 1e-4
        y_pred = self.xp.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 1])

        self.layer.backward(y_pred, y_true)
        samples = y_pred.shape[0]

        def scalar_loss(preds):
            # Sum of per-sample losses, matching the un-normalized quantity
            # dinputs was derived from (divided by `samples` in backward).
            return self.xp.sum(self.layer.forward(preds, y_true, training=False))

        numerical = self.xp.zeros_like(y_pred)
        B, C = y_pred.shape
        for b in range(B):
            for c in range(C):
                plus = y_pred.copy()
                minus = y_pred.copy()
                plus[b, c] += epsilon
                minus[b, c] -= epsilon

                loss_plus = scalar_loss(plus)
                loss_minus = scalar_loss(minus)

                numerical[b, c] = (loss_plus - loss_minus) / (2 * epsilon)

        numerical /= samples

        self.xp.testing.assert_array_almost_equal(
            self.layer.dinputs, numerical, decimal=3,
            err_msg="Analytical backward pass does not match the numerical gradient of forward()."
        )

    def test_backward_numerical_gradient_check_with_label_smoothing(self):
        """Same finite-difference check, but with label smoothing engaged on both passes."""
        epsilon = 1e-4
        smoothed = self.make_component(
            CategoricalCrossEntropy, 
            label_smoothing=0.1,
        )

        y_pred = self.xp.array([
            [0.55, 0.35, 0.10],
            [0.15, 0.65, 0.20],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 1])

        smoothed.backward(y_pred, y_true)
        samples = y_pred.shape[0]

        def scalar_loss(preds):
            # training=True so forward() applies the same smoothing as backward()
            return self.xp.sum(smoothed.forward(preds, y_true, training=True))

        numerical = self.xp.zeros_like(y_pred)
        B, C = y_pred.shape
        for b in range(B):
            for c in range(C):
                plus = y_pred.copy()
                minus = y_pred.copy()
                plus[b, c] += epsilon
                minus[b, c] -= epsilon

                loss_plus = scalar_loss(plus)
                loss_minus = scalar_loss(minus)

                numerical[b, c] = (loss_plus - loss_minus) / (2 * epsilon)

        numerical /= samples

        self.xp.testing.assert_array_almost_equal(
            smoothed.dinputs, numerical, decimal=3,
            err_msg="Backward pass with label smoothing does not match the numerical gradient."
        )

    def test_backward_clipping_prevents_div_by_zero(self):
        """Predictions of exactly 0 or 1 should not produce inf/nan in dinputs."""
        y_pred = self.xp.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 2])

        self.layer.backward(y_pred, y_true)

        self.assertFalse(self.xp.isnan(self.layer.dinputs).any())
        self.assertFalse(self.xp.isinf(self.layer.dinputs).any())

    def test_backward_does_not_mutate_dvalues(self):
        """Backward pass should not alter the incoming predictions tensor in-place."""
        y_pred = self.xp.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 2])
        original = y_pred.copy()

        self.layer.backward(y_pred, y_true)

        self.xp.testing.assert_array_equal(
            original, y_pred,
            err_msg="Backward pass mutated dvalues in-place."
        )

    # ---- Accumulation (calculate / calculate_accumulated / new_pass) ----

    def test_calculate_matches_forward_mean(self):
        y_pred = self.xp.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
        ], dtype=self.xp.float32)
        y_true = self.xp.array([0, 1])

        self.layer.new_pass()
        calculated = self.layer.calculate(y_pred, y_true)
        expected = self.xp.mean(self.layer.forward(y_pred, y_true, training=True))

        self.assertAlmostEqual(float(calculated), float(expected), places=5)

    def test_calculate_accumulates_across_batches(self):
        """calculate_accumulated should reflect the running mean across multiple calculate() calls."""
        self.layer.new_pass()

        y_pred_1 = self.xp.array([[0.8, 0.1, 0.1]], dtype=self.xp.float32)
        y_true_1 = self.xp.array([0])
        y_pred_2 = self.xp.array([[0.2, 0.2, 0.6]], dtype=self.xp.float32)
        y_true_2 = self.xp.array([2])

        loss_1 = self.layer.calculate(y_pred_1, y_true_1)
        loss_2 = self.layer.calculate(y_pred_2, y_true_2)

        accumulated = self.layer.calculate_accumulated()
        expected = (loss_1 + loss_2) / 2

        self.assertAlmostEqual(float(accumulated), float(expected), places=5)

    def test_new_pass_resets_accumulators(self):
        y_pred = self.xp.array([[0.8, 0.1, 0.1]], dtype=self.xp.float32)
        y_true = self.xp.array([0])

        self.layer.new_pass()
        self.layer.calculate(y_pred, y_true)

        self.layer.new_pass()

        self.assertEqual(self.layer.accumulated_sum, 0)
        self.assertEqual(self.layer.accumulated_count, 0)

    # ---- Regularization ---------------------

    def test_regularization_loss_known_values(self):
        """One regularizer per layer, weights [1, 2] and biases [3, -1] each, so dropping
        any of the four terms changes the total; the unregularized layer adds nothing."""
        regularizers = [
            {"l1": (0.2,)},         # 0.2 * (1 + 2)                = 0.6
            {"l2": (0.1,)},         # 0.5 * 0.1 * (1 + 4)          = 0.25
            {"l1": (0.0, 0.3)},     # 0.3 * (3 + 1)                = 1.2
            {"l2": (0.0, 0.4)},     # 0.5 * 0.4 * (9 + 1)          = 2.0
            {},                     # unregularized                = 0
        ]
        layers = []
        for kwargs in regularizers:
            layer = self.make_built_layer(Dense, input_shape=(1,), n_inputs=1, n_neurons=2, **kwargs)
            layer.weights = self.xp.array([[1.0, 2.0]], dtype=self.xp.float32)
            layer.biases = self.xp.array([[3.0, -1.0]], dtype=self.xp.float32)
            layers.append(layer)
        layers[-1].weights = self.xp.array([[100.0, 100.0]], dtype=self.xp.float32)

        self.layer.remember_trainable_layers(layers)

        self.assertAlmostEqual(float(self.layer.regularization_loss()), 4.05, places=5)


base_case.register_test_suites(globals(), TestLossCCE)