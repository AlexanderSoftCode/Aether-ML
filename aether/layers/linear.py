import math
import numpy as np
import aether.config as config
from aether.base import Layer

class Dense(Layer):
    """Fully connected (Dense) layer.

    Parameters
    ----------
    n_inputs : int
        Number of input features.
    n_neurons : int
        Number of output neurons.
    l1 : float or tuple, default=()
        L1 penalty. Pass a float for weights only, or (weight, bias).
    l2 : float or tuple, default=()
        L2 penalty. Pass a float for weights only, or (weight, bias).

    Notes
    -----
    Expected input shape: ``(batch_size, n_inputs)``
    Output shape: ``(batch_size, n_neurons)``
    """

    def __init__(self, n_inputs, n_neurons, l1=(), l2=()):
        self.n_inputs = n_inputs
        self.n_neurons= n_neurons
        self._set_regularizers(l1, l2)

        self.seed = None
        self.weights = None
        self.biases = None

        self.precision_policy = config.DTypePolicy(compute_dtype = None)

        # Ephemeral forward cache for backward reuse
        # Is also not used during inference
        self._inputs_compute = None
        self._weights_compute = None

        # Shadow cast of weights/biases at compute precision, reused across
        # forward calls until the weights, biases, or precision policy change.
        self._weights_compute_cache = None
        self._biases_compute_cache = None
        self._compute_cache_valid = False

    def build(self, input_shape: tuple[int, ...], seed: int | None = None) -> tuple[int, ...]:
        """
        Called once by Model.finalize(). config.xp is guaranteed to be
        correctly set if the user called model.to() beforehand.
        """
        super().build(input_shape)
        xp = config.xp

        
        if self.n_inputs != input_shape[0]:
            raise ValueError(
                f"[aether] Shape mismatch in Dense layer: configured with n_inputs={self.n_inputs}, "
                f"but received input shape with dimension {input_shape[0]} ({input_shape})."
            )

        self.input_shape = input_shape

        
        std = (2.0 / self.n_inputs) ** 0.5

        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            self.seed = effective_seed
            rng = xp.random.RandomState(effective_seed)
            raw_weights = rng.randn(self.n_inputs, self.n_neurons)
        else:
            raw_weights = xp.random.randn(self.n_inputs, self.n_neurons)

        self.weights = (raw_weights * std).astype(xp.float32, copy=False)
        self.biases = xp.zeros((1, self.n_neurons), dtype=xp.float32)
        self._compute_cache_valid = False

        self.output_shape = (self.n_neurons,)
        return self.output_shape

    def _apply_precision(self, policy):
        """
        Called on Model.set_precision(), stores current policy and invalidates shadow
        cache. All astype checks happen on forward.
        """
        self.precision_policy = policy or config.DTypePolicy()
        self._compute_cache_valid = False

    def invalidate_shadow_caches(self):
        """
        Called by an optimizer after it writes into self.weights/self.biases'
        underlying buffers in place (e.g. a fused GPU kernel taking the array
        as an output pointer) -- that kind of update never goes through a
        property setter, so _compute_cache_valid would otherwise stay True
        and forward() would keep matmul-ing against a stale compute-precision
        snapshot. Forces a rebuild on the next forward() that needs it.
        """
        self._compute_cache_valid = False

    def forward(self, inputs, training):
        xp = config.get_array_module(inputs)

        inputs_c = self.precision_policy.cast_to_compute(inputs)

        if not self._compute_cache_valid:
            self._weights_compute_cache, self._biases_compute_cache = self.precision_policy.cast_to_compute(
                self.weights, self.biases
            )
            self._compute_cache_valid = True

        weights_c = self._weights_compute_cache
        biases_c = self._biases_compute_cache

        if training:
            self.inputs = inputs
            self._inputs_compute = inputs_c
            self._weights_compute = weights_c
        else:
            self.inputs = None
            self._inputs_compute = None
            self._weights_compute = None

        return xp.dot(inputs_c, weights_c) + biases_c 
    
    def backward(self, dvalues):

        xp = config.get_array_module(dvalues)

        if self._inputs_compute is None or self._weights_compute is None:
            raise RuntimeError(
                "Dense.backard() called without a preceeding training=True during" \
                "forward pass. If you are manually tracing through the layers, please" \
                " add training=True to Dense.forward()"
            )
        dvalues_c = self.precision_policy.cast_to_compute(dvalues)

        #Fallback, rederive from self.inputs/self.weights rather then crashing
        inputs_c = self._inputs_compute if self._inputs_compute is not None \
        else self.precision_policy.cast_to_compute(self.inputs)[0]
        weights_c = self._weights_compute if self._weights_compute is not None \
        else self.precision_policy.cast_to_compute(self.weights)[0]

        dweights = xp.dot(inputs_c.T, dvalues)
        dbiases = xp.sum(dvalues, axis = 0, keepdims = True)
        dinputs_c = xp.dot(dvalues_c, weights_c.T)

        self.dweights, self.dbiases = self.precision_policy.cast_to_param(dweights, dbiases)
        self.dinputs = dinputs_c.astype(self.inputs.dtype, copy=False)

        self._inputs_compute = None
        self._weights_compute = None

        return self.dinputs

    def get_config(self) -> dict:
        return {
            "n_inputs": self.n_inputs,
            "n_neurons": self.n_neurons,
            "l1": (self.weight_regularizer_l1, self.bias_regularizer_l1),
            "l2": (self.weight_regularizer_l2, self.bias_regularizer_l2),
        }
    def get_parameters(self) -> dict:
        return {"weights": self.weights, "biases": self.biases}

    def set_parameters(self, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        if biases is not None:
            self.biases = biases
        self._compute_cache_valid = False

class Flatten(Layer):
    """Flattens all trailing input dimensions into a 2D tensor
    while preserving the batch dimension.
    
    Notes
    -----
    Expected input shape: ``(batch_size, d_1, d_2, ..., d_n)``
    Output shape: ``(batch_size, d_1 * d_2 * ... * d_n)``
    """
    def build(self, input_shape: tuple[int, ...], seed: int | None = None) -> tuple[int, ...]:    
        super().build(input_shape)
        self.input_shape = input_shape 
        
        # Calculate scalar integer dimension using standard library math.prod
        flat_dim = math.prod(input_shape)
        
        self.output_shape = (flat_dim,)
        return self.output_shape

    def forward(self, inputs, training):
        xp = config.get_array_module(inputs)

        return xp.ascontiguousarray(inputs.reshape(inputs.shape[0], -1))
    
    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(-1, *self.input_shape)
        return self.dinputs