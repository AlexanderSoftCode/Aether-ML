from tests.base_case import register_test_suites
from tests.integration.model_base_suite import ModelIntegrationBaseCase


class TestCnnTrainingPipeline(ModelIntegrationBaseCase):
    N_SAMPLES = 16
    BATCH_SIZE = 4

    def setUp(self):
        super().setUp()
        self.X, self.y = self.make_synthetic_image_data(n_samples=self.N_SAMPLES)

    def test_train_runs_and_produces_finite_loss(self):
        model = self.build_cnn_model()
        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )
        val_loss, val_acc = model.evaluate(
            self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False
        )

        self.assertTrue(bool(self.xp.isfinite(val_loss)))
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 1.0)

    def test_gradients_reach_conv_and_dense_layers(self):
        """White-box check: verifies gradients propagate through all trainable
        Conv and Dense layers using raw unnormalized logits (matching Model.train)."""
        model = self.build_cnn_model()

        # 1. Forward pass returns raw logits
        out = model.forward(self.X, training=True)
        model.loss.calculate(out, self.y)

        # 2. Pass 'out' (raw logits) directly into fused loss backward
        model.loss.backward(out, self.y)
        model.backward(model.loss.dinputs)

        # 3. Verify gradients on all trainable layers (Conv, Dense, etc.)
        for layer in model.trainable_layers:
            with self.subTest(layer=type(layer).__name__):
                self.assertTrue(hasattr(layer, "dweights"))
                self.assertEqual(layer.dweights.shape, layer.weights.shape)
                self.assertTrue(bool(self.xp.all(self.xp.isfinite(layer.dweights))))

                if hasattr(layer, "biases") and layer.biases is not None:
                    self.assertTrue(hasattr(layer, "dbiases"))
                    self.assertEqual(layer.dbiases.shape, layer.biases.shape)
                    self.assertTrue(bool(self.xp.all(self.xp.isfinite(layer.dbiases))))

    def test_predict_batched_matches_unbatched(self):
        model = self.build_cnn_model()
        full = model.predict(self.X, batch_size=None)
        batched = model.predict(self.X, batch_size=self.BATCH_SIZE)
        self.xp.testing.assert_allclose(full, batched, rtol=1e-2, atol=1e-2)

    def test_fp16_precision_forward_produces_finite_output_on_gpu(self):
        """confirms the composition survives set_precision('float16') end-to-end through
        Dense + Conv together, not that Conv's behavior changes because of it."""
        if self.backend_name != "cupy":
            self.skipTest(
                "fp16 compute path is only meaningfully exercised on the CuPy backend."
            )

        model = self.build_cnn_model(precision="float16")
        out = model.forward(self.X, training=False)
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(out))))

    def test_spatial_dropout_changes_output_between_train_and_eval(self):
        """Sanity check that training=True/False actually routes through
        SpatialDropout differently."""
        model = self.build_cnn_model()

        train_out_1 = model.forward(self.X, training=True).copy()
        model._rng_clock.advance()
        train_out_2 = model.forward(self.X, training=True)

        eval_out_1 = model.forward(self.X, training=False).copy()
        eval_out_2 = model.forward(self.X, training=False)

        # Successive training steps draw fresh masks.
        self.assertFalse(bool(self.xp.allclose(train_out_1, train_out_2)))

        # Eval is deterministic and mask-free.
        self.xp.testing.assert_allclose(eval_out_1, eval_out_2, rtol=1e-5, atol=1e-6)

        # Dropout is actually active in training mode.
        self.assertFalse(bool(self.xp.allclose(train_out_1, eval_out_1)))

    def test_loss_decreases_over_training_steps(self):
        """Composition-level check driven through the public Model.train()
        API on a small fixed batch."""
        model = self.build_cnn_model()

        model.train(
            X=self.X,
            y=self.y,
            epochs=1,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )
        first_loss, _ = model.evaluate(
            self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False
        )

        model.train(
            X=self.X,
            y=self.y,
            epochs=15,
            batch_size=self.BATCH_SIZE,
            verbose=False,
            print_every=10**9,
        )
        later_loss, _ = model.evaluate(
            self.X, self.y, batch_size=self.BATCH_SIZE, verbose=False
        )

        self.assertLess(later_loss, first_loss)


register_test_suites(globals(), TestCnnTrainingPipeline)