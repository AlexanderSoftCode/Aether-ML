# tests/model/base.py
import numpy as np
import aether.config as config
from tests.base_case import AetherBaseTestCase

from aether.base import Layer
from aether.losses import Loss
from aether.optimizers import Optimizer
from aether.metrics import Accuracy


class SpyLifecycleLayer(Layer):
    """Spy layer for tracking device compilation, seed offsets, and precision dispatch."""
    def __init__(self, n_inputs=4, n_neurons=3, precision_exempt=False):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self._precision_exempt = precision_exempt

        self.is_built = False
        self.weights = None
        self.biases = None
        self.dweights = None
        self.dbiases = None
        self.compiled_device = None
        self.applied_policy = None
        self.seed = None

    def build(self, input_shape: tuple[int, ...], seed: int | None = None) -> tuple[int, ...]:
        self.seed = seed
        xp = config.xp
        self.weights = xp.zeros((self.n_inputs, self.n_neurons), dtype=xp.float32)
        self.biases = xp.zeros((1, self.n_neurons), dtype=xp.float32)
        self.is_built = True
        return (self.n_neurons,)

    def forward(self, X, training=True):
        xp = config.get_array_module(X)
        return xp.zeros((X.shape[0], self.n_neurons), dtype=X.dtype)

    def backward(self, dvalues):
        xp = config.get_array_module(dvalues)
        self.dweights = xp.zeros((self.n_inputs, self.n_neurons), dtype=dvalues.dtype)
        self.dbiases = xp.zeros((1, self.n_neurons), dtype=dvalues.dtype)
        return xp.zeros((dvalues.shape[0], self.n_inputs), dtype=dvalues.dtype)

    def _compile_for_device(self, device):
        self.compiled_device = device

    def _apply_precision(self, policy):
        self.applied_policy = policy


class SpySetSeedLayer(Layer):
    """Spy layer relying strictly on the _set_seed() hook instead of build()."""
    is_stochastic=True
    def __init__(self):
        super().__init__()
        self.seed = None
        self._seed_key = None
        self._clock = None
        self.bind_calls = []
        
    def _bind_rng(self, *, base_seed, stream_id, clock):
        self.bind_calls.append((base_seed, stream_id))
        base = self.seed if self.seed is not None else base_seed
        self._seed_key = config.derive_stream_seed(base, stream_id)
        self._clock = clock

    def _set_seed(self, seed):
        self.seed = seed

    def forward(self, X, training=True):
        return X

    def backward(self, dvalues):
        return dvalues


class SpySyncLoss(Loss):
    def __init__(self):
        super().__init__()
        self.compiled_device = None

    def _compile_for_device(self, device):
        self.compiled_device = device

    def forward(self, y_pred, y_true):
        return config.xp.array(0.0)

    def backward(self, dvalues, y_true):
        self.dinputs = dvalues


class SpySyncOptimizer(Optimizer):
    def __init__(self):
        self.compiled_device = None

    def _compile_for_device(self, device):
        self.compiled_device = device

    def init_params(self, trainable_layers):
        pass

    def step(self):
        pass


class SpySyncAccuracy(Accuracy):
    def __init__(self):
        super().__init__()
        self.compiled_device = None

    def _compile_for_device(self, device):
        self.compiled_device = device

    def compare(self, predictions, y):
        return config.xp.array(True)


class ModelBaseTestCase(AetherBaseTestCase):
    """Shared base TestCase providing synthetic datasets and mock components."""
    __test__ = False

    NUM_SAMPLES = 32
    NUM_FEATURES = 4
    NUM_CLASSES = 3

    def setUp(self):
        super().setUp()
        self.xp.random.seed(42)
        np.random.seed(42)

        self.X = self.xp.random.randn(self.NUM_SAMPLES, self.NUM_FEATURES).astype("float32")
        self.y = self.xp.random.randint(0, self.NUM_CLASSES, size=(self.NUM_SAMPLES,)).astype("int32")

        self.X_val = self.xp.random.randn(16, self.NUM_FEATURES).astype("float32")
        self.y_val = self.xp.random.randint(0, self.NUM_CLASSES, size=(16,)).astype("int32")