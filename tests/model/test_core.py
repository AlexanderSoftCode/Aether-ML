from tests.base_case import register_test_suites
from tests.model.base import ModelBaseTestCase, SpyLifecycleLayer, SpySetSeedLayer
from aether.model import Model
from aether.layers.linear import Dense
from aether.layers.activations import SoftMax
from aether.losses import SoftmaxCategoricalCrossEntropy
from aether.optimizers import Adam
from aether._utils import NullAccuracy, NullOptimizer


class TestModelCore(ModelBaseTestCase):
    __test__ = False

    def test_add_layer_and_type_validation(self):
        """Ensure only instances of Layer can be added and layer count updates."""
        model = Model()
        layer = Dense(self.NUM_FEATURES, 8)
        model.add(layer)
        self.assertEqual(len(model.layers), 1)
        self.assertIs(model.layers[0], layer)

        with self.assertRaises(TypeError):
            model.add("InvalidLayerObject")

    def test_deferred_build_allocation_on_finalize(self):
        """Verify parameters remain unallocated until finalize() explicitly triggers build()."""
        model = Model()
        layer1 = SpyLifecycleLayer(self.NUM_FEATURES, 8)
        layer2 = SpyLifecycleLayer(8, self.NUM_CLASSES)

        model.add(layer1)
        model.add(layer2)

        self.assertFalse(layer1.is_built)
        self.assertIsNone(layer1.weights)
        self.assertIsNone(layer1.biases)
        self.assertFalse(layer2.is_built)
        self.assertIsNone(layer2.weights)
        self.assertIsNone(layer2.biases)

        model.finalize((self.NUM_FEATURES,))

        self.assertTrue(layer1.is_built)
        self.assertIsNotNone(layer1.weights)
        self.assertIsNotNone(layer1.biases)
        self.assertEqual(layer1.weights.shape, (self.NUM_FEATURES, 8))
        self.assertEqual(layer1.biases.shape, (1, 8))

        self.assertTrue(layer2.is_built)
        self.assertIsNotNone(layer2.weights)
        self.assertIsNotNone(layer2.biases)
        self.assertEqual(layer2.weights.shape, (8, self.NUM_CLASSES))
        self.assertEqual(layer2.biases.shape, (1, self.NUM_CLASSES))

        self.assertEqual(len(model.trainable_layers), 2)
        self.assertIn(layer1, model.trainable_layers)
        self.assertIn(layer2, model.trainable_layers)

    def test_manual_seed_sets_in_place_and_post_finalize_guard(self):
        """Test Model.manual_seed() mutates in place (no chaining) and ensure
        post-finalize modifications raise."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 8))

        self.assertIsNone(model.manual_seed(1234))
        self.assertEqual(model._seed, 1234)

        model.finalize((self.NUM_FEATURES,))

        with self.assertRaises(RuntimeError):
            model.manual_seed(5678)

    def test_seed_propagation_to_layers(self):
        """Verify model.finalize() distributes indexed seeds deterministically."""
        base_seed = 100

        model1 = Model()
        l1_m1 = SpyLifecycleLayer(self.NUM_FEATURES, 8)
        l2_m1 = SpyLifecycleLayer(8, self.NUM_CLASSES)
        l3_m1 = SpySetSeedLayer()
        model1.manual_seed(base_seed)
        model1.add(l1_m1)
        model1.add(l2_m1)
        model1.add(l3_m1)
        model1.finalize((self.NUM_FEATURES,))

        model2 = Model()
        l1_m2 = SpyLifecycleLayer(self.NUM_FEATURES, 8)
        l2_m2 = SpyLifecycleLayer(8, self.NUM_CLASSES)
        l3_m2 = SpySetSeedLayer()
        model2.manual_seed(base_seed)
        model2.add(l1_m2)
        model2.add(l2_m2)
        model2.add(l3_m2)
        model2.finalize((self.NUM_FEATURES,))

        self.assertEqual(l1_m1.seed, base_seed + 0)
        self.assertEqual(l2_m1.seed, base_seed + 1)

        self.assertIsNone(l3_m1.seed)
        self.assertEqual(l3_m1._seed_key, l3_m2._seed_key)
        self.assertIs(l3_m1._clock, model1._rng_clock)

        self.xp.testing.assert_allclose(l1_m1.weights, l1_m2.weights)
        self.xp.testing.assert_allclose(l2_m1.weights, l2_m2.weights)

    def test_mutation_after_finalize_raises(self):
        """Ensure graph mutations are locked once finalize() has been called."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 8))
        model.finalize((self.NUM_FEATURES,))

        with self.assertRaises(RuntimeError):
            model.add(Dense(8, 2))

        with self.assertRaises(RuntimeError):
            model.finalize((self.NUM_FEATURES,))

    def test_finalize_empty_model_raises(self):
        """Verify that finalizing an empty model raises a RuntimeError."""
        model = Model()
        with self.assertRaises(RuntimeError):
            model.finalize((self.NUM_FEATURES,))

    def test_finalize_defaults_null_optimizer_and_accuracy(self):
        """Ensure NullOptimizer and NullAccuracy placeholders are bound if left unconfigured."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, self.NUM_CLASSES))
        model.finalize((self.NUM_FEATURES,))

        self.assertIsInstance(model.optimizer, NullOptimizer)
        self.assertIsInstance(model.accuracy, NullAccuracy)

    def test_softmax_cce_fusion_validation(self):
        """Verify finalize() raises ValueError when a trailing SoftMax is added alongside SoftmaxCategoricalCrossEntropy."""
        model = Model()
        model.add(Dense(self.NUM_FEATURES, 8))
        model.add(Dense(8, self.NUM_CLASSES))
        model.add(SoftMax())
        model.configure(loss=SoftmaxCategoricalCrossEntropy())

        with self.assertRaises(ValueError):
            model.finalize((self.NUM_FEATURES,))


register_test_suites(globals(), TestModelCore)