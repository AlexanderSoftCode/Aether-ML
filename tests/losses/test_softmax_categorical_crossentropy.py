import tests.base_case as base_case
from aether.layers.activations import SoftMax
from aether.layers.linear import Dense
from aether.losses.categorical_crossentropy import (
    CategoricalCrossEntropy,
    SoftmaxCategoricalCrossEntropy,
)


class TestActivationSoftmaxLossCCE(base_case.AetherBaseTestCase):
    LABEL_SMOOTHING = 0.1 

    def setUp(self):
        super().setUp()
        self.layer = self.make_component(
            SoftmaxCategoricalCrossEntropy, 
            label_smoothing=self.LABEL_SMOOTHING,
        )
        self.layer.new_pass()

    def test_output_matches_softmax_forward(self):
        logits = self.xp.array([
            [3.0, 2.0, 1.0, 1.5],
            [-3.0, 0.5, 2.5, 1.0]
        ])

        y_true = self.xp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        layer_softmax = self.make_built_layer(
            SoftMax, input_shape=(4,),
        )
        layer_cce_loss = self.make_component(
            CategoricalCrossEntropy, 
            label_smoothing=self.LABEL_SMOOTHING,
        )
        layer_cce_loss.new_pass()
        
        softmax_output = layer_softmax.forward(logits, training=True)
        desired = layer_cce_loss.calculate(softmax_output, y_true, training=True)

        # Reduce the per-sample loss array to scalar mean for comparison with .calculate()
        expected = self.xp.mean(self.layer.forward(logits, y_true, training=True))

        self.xp.testing.assert_allclose(desired, expected, rtol=1e-4)


    def test_forward_matches_separate_softmax_and_cce(self):
        """Verify calculating loss in one combined step mimics the same result
        as calculating the result in two separate steps:
        Combined Forward Loss (X, y) = Loss(Softmax(X), y)"""
        logits = self.xp.array([
            [3.0, 2.0, 1.0, 1.5],
            [-3.0, 0.5, 2.5, 1.0]
        ])

        y_true = self.xp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])
        layer_softmax = self.make_component(
            SoftMax,
        )
        layer_cce_loss = self.make_component(
            CategoricalCrossEntropy, 
            label_smoothing=self.LABEL_SMOOTHING,
        )
        
        softmax_output = layer_softmax.forward(logits, training=True)
        expected_separate_output = layer_cce_loss.calculate(softmax_output, y_true, training=True)
        
        # Reduce per-sample loss vector to scalar mean
        combined_output = self.xp.mean(self.layer.forward(logits, y_true, training=True))
        
        self.xp.testing.assert_allclose(expected_separate_output, combined_output, rtol=1e-4)

    def test_forward_numerical_stability(self):
        """ Verify large-magnitude logits (e.g +- 1000) do not produce Nan/Inf
        in returned loss, for the scaler loss."""

        logits = self.xp.array([
            [1000.0, 2000.0, 0.0, -500.0],
            [10.0, 500.0, 1.3, -35.0]
        ])
        y_true = self.xp.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        expected = self.layer.forward(logits, y_true, training=True)
        finite_mask = self.xp.isfinite(expected)
        all_finite = self.xp.all(finite_mask)
        self.assertTrue(all_finite)

    def test_backward_gradient_shape(self):
        """Verify shape matches the input logits shape"""
        logits = self.xp.array([
            [3.0, 2.0, 1.0, 1.5],
            [-3.0, 0.5, 2.5, 1.0]
        ])
        y_true = self.xp.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

        self.layer.forward(logits, y_true, training=True)
        self.layer.backward(logits, y_true)

        self.assertEqual(self.layer.dinputs.shape, logits.shape)

    def test_backward_matches_shortcut_formula(self):
        """Verify combined backwards dinputs equates to 
        (probs - y_true_onehot) / samples , exactly"""

        logits = self.xp.array([
            [-1.2,  4.5,  0.8, -0.5],
            [ 2.1, -1.0, -3.4,  3.8]
        ])

        y_true = self.xp.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        # Find expected dinputs
        self.layer.forward(logits, y_true, training=True)
        probs = self.layer.activation.output
        self.layer.backward(logits, y_true)
        expected_dinputs = self.layer.dinputs

        # Calculate desired dinputs accounting for label smoothing
        ls = self.layer.label_smoothing
        num_samples, n_classes = logits.shape
        y_true_smooth = y_true * (1.0 - ls) + (ls / n_classes)

        # Compare results 
        desired_dinputs = (probs - y_true_smooth) / num_samples
        self.xp.testing.assert_allclose(expected_dinputs, desired_dinputs, rtol=1e-4)

    def test_backward_matches_two_step_chain(self):
        """ Verify the combined loss classes dinputs matches the backwards pass
        of Softmax.backward(...)"""

        logits = self.xp.array([
            [-1.2,  4.5,  0.8, -0.5],
            [ 2.1, -1.0, -3.4,  3.8]
        ])

        y_true = self.xp.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        layer_softmax = self.make_component(
            SoftMax,
        )
        layer_cce_loss = self.make_component(
            CategoricalCrossEntropy, 
            label_smoothing=self.LABEL_SMOOTHING,
        )

        layer_softmax.forward(logits, training=True)
        layer_cce_loss.calculate(layer_softmax.output, y_true, training=True)
        layer_cce_loss.backward(layer_softmax.output, y_true)
        layer_softmax.backward(layer_cce_loss.dinputs)
        actual_dinputs = layer_softmax.dinputs

        self.layer.forward(logits, y_true, training=True)
        self.layer.backward(logits, y_true)
        expected_dinputs = self.layer.dinputs
        self.xp.testing.assert_allclose(actual_dinputs, expected_dinputs, rtol=1e-4)

    def test_backward_does_not_mutate_dvalues(self):
        """Confirm the method does not modify logits in place"""

        logits = self.xp.array([
            [-1.2,  4.5,  0.8, -0.5],
            [ 2.1, -1.0, -3.4,  3.8]
        ], dtype=self.xp.float32)

        y_true = self.xp.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=self.xp.float32)

        self.layer.forward(logits, y_true, training=True)
        logits_snapshot = logits.copy()
        self.layer.backward(logits, y_true)

        self.xp.testing.assert_array_equal(
            logits_snapshot, logits,
            err_msg='logits was mutated in place'
        )

    def test_backward_sparse_and_onehot_labels_agree(self): 
        """dinputs identical regardless of label format"""

        logits = self.xp.array([
            [-1.2,  4.5,  0.8, -0.5],
            [ 2.1, -1.0, -3.4,  3.8]
        ], dtype=self.xp.float32)

        y_true_onehot = self.xp.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=self.xp.float32)
        y_true_sparse = self.xp.array([1, 3], dtype=self.xp.int32)

        # One hot version
        layer_onehot = self.make_component(
            SoftmaxCategoricalCrossEntropy, 
            label_smoothing=self.LABEL_SMOOTHING,
        )
        layer_onehot.forward(logits, y_true_onehot, training=True)
        layer_onehot.backward(logits, y_true_onehot)
        dinputs_onehot = layer_onehot.dinputs.copy()

        # Sparse version
        layer_sparse = self.make_component(
            SoftmaxCategoricalCrossEntropy, 
            label_smoothing=self.LABEL_SMOOTHING,
        )
        layer_sparse.forward(logits, y_true_sparse, training=True)
        layer_sparse.backward(logits, y_true_sparse)
        dinputs_sparse = layer_sparse.dinputs.copy()

        self.xp.testing.assert_allclose(dinputs_onehot, dinputs_sparse)
        
    def test_single_sample_batch(self):
        """Verify forward and backward passes work for batch_size = 1"""

        logits = self.xp.array([
            [-1.2,  4.5,  0.8, -0.5]
        ], dtype=self.xp.float32)

        y_true = self.xp.array([
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=self.xp.float32)

        loss = self.layer.forward(logits, y_true, training=True)
        self.assertEqual(self.layer.activation.output.shape, (1, 4))
        self.assertFalse(self.xp.isnan(loss).any())

        self.layer.backward(logits, y_true)
        self.assertEqual(self.layer.dinputs.shape, (1, 4))
        self.assertFalse(self.xp.isnan(self.layer.dinputs).any())

    def test_backward_numerical_check(self):
        """Numerical gradient check for standard Softmax + CCE (label_smoothing = 0.0)"""
        layer = self.make_component(
            SoftmaxCategoricalCrossEntropy, 
            label_smoothing=0.0,
        )

        logits = self.xp.array([
            [-1.2,  4.5,  0.8, -0.5],
            [ 2.1, -1.0, -3.4,  3.8]
        ], dtype=self.xp.float32)

        y_true = self.xp.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=self.xp.float32)

        layer.forward(logits, y_true, training=True)
        layer.backward(logits, y_true)
        analytical_dinputs = layer.dinputs.copy()

        h = 1e-4
        num_dinputs = self.xp.zeros_like(logits)

        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                logits_plus = logits.copy()
                logits_plus[i, j] += h
                loss_plus = self.xp.mean(layer.forward(logits_plus, y_true, training=False))

                logits_minus = logits.copy()
                logits_minus[i, j] -= h
                loss_minus = self.xp.mean(layer.forward(logits_minus, y_true, training=False))

                num_dinputs[i, j] = (loss_plus - loss_minus) / (2 * h)

        self.xp.testing.assert_allclose(
            analytical_dinputs, 
            num_dinputs, 
            rtol=1e-3, 
            atol=1e-3,
            err_msg="Analytical and numerical gradients do not match for standard CCE."
        )

    def test_backward_numerical_check_label_smoothing(self):
        """Check finite-difference forward() wrt. raw logits; compare to dinputs"""
        epsilon = 1e-4
        logits = self.xp.array([
            [-1.2,  4.5,  0.8, -0.5],
            [ 2.1, -1.0, -3.4,  3.8]
        ], dtype=self.xp.float32)

        y_true = self.xp.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=self.xp.float32)

        _ = self.layer.forward(logits, y_true, training=True)
        self.layer.backward(logits, y_true, training=True)
        analytical_dinputs = self.layer.dinputs

        num_dinputs = self.xp.zeros_like(logits)
        
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                logits_plus = logits.copy()
                logits_plus[i, j] += epsilon
                loss_plus = self.xp.mean(self.layer.forward(logits_plus, y_true, training=True))

                logits_minus = logits.copy()
                logits_minus[i, j] -= epsilon
                loss_minus = self.xp.mean(self.layer.forward(logits_minus, y_true, training=True))

                num_dinputs[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        self.xp.testing.assert_allclose(
            analytical_dinputs, 
            num_dinputs, 
            rtol=1e-3, 
            atol=1e-3,
            err_msg="Analytical dinputs do not match numerical finite difference gradients!"
        )
    def test_combined_loss_raises_error_for_layer_when_softmax(self):
        last_layer_in_network = self.make_built_layer(
            SoftMax, input_shape=(4,)
        )
        
        with self.assertRaises(ValueError) as ctx:
            self.layer.validate_graph(layers=[last_layer_in_network])
            
        # Check for key keywords rather than exact long phrases
        self.assertIn("unnormalized", str(ctx.exception))
        self.assertIn("SoftMax", str(ctx.exception))

    def test_combined_loss_passes_for_valid_preceding_layer(self):
        """Confirms validate_graph succeeds when the preceding layer is not prohibited."""
        valid_preceding_layer = self.make_built_layer(
            Dense, n_inputs=4, n_neurons=4, input_shape=(4,)
        )
        
        try:
            self.layer.validate_graph(layers=[valid_preceding_layer])
        except ValueError:
            self.fail("validate_graph raised ValueError unexpectedly on a valid layer!")


base_case.register_test_suites(globals(), TestActivationSoftmaxLossCCE)