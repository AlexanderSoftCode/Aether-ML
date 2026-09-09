import numpy as np 
import aether.config as config

class Layer:
    """Abstract base class defining shape-inference, parameter routing, and
    state manager for all network layers."""
    _precision_exempt: bool = False
    is_stochastic: bool = False
    def __init__(self):
        self.seed = None
        # these attributes below will likely be used for Model.save()
        self.input_shape = None
        self.output_shape = None
        self.weights = None 
        self.biases = None
        self.precision_policy = None

    def _apply_precision(self, policy):
        self.precision_policy = policy
    def build(self, input_shape, seed=None):
        """
        Base build contract: stores shapes and passes through.
        Does not mutate self.seed on stateless layers.
        """
        self.input_shape = input_shape
        self.output_shape = input_shape

        return self.output_shape
    
    def get_config(self) -> dict:
        """Override to return constructor kwargs required to reconstruct the layer. Used by Model.load()"""
        return {}

    def _resolve_compute_dtype(self, xp):
        """
        Resolves against a live xp handle the caller already has
        EX: `xp = config.get_array_module(inputs)` inside forward/backward
        Layer deliberately never caches its own xp refernce
        """

        name = self.precision_policy.compute_dtype_name if self.precision_policy else 'float32'
        return xp.dtype(name)
    
    def _compile_for_device(self, device):
        """
        Hook for layers utilizing fused modules to 
        re-bind runtime pointers
        """
        pass

    def get_parameters(self) -> dict[str, object]:
        """Returns a dict of state tensors belonging to this layer."""
        params = {}
        if self.weights is not None:
            params["weights"] = self.weights
        if self.biases is not None:
            params["biases"] = self.biases
        return params

    def set_parameters(self, **params):
        """Sets state tensors to this layer."""
        for name, value in params.items():
            if hasattr(self, name):
                setattr(self, name, value)

    @staticmethod
    def _parse_reg(reg):
        """Normalizes scalar, 1-tuple, or 2-tuple into (weight_reg, bias_reg)."""
        if isinstance(reg, (int, float)):
            return (float(reg), 0.0)
        if not reg:
            return (0.0, 0.0)
        if len(reg) == 1:
            return (float(reg[0]), 0.0)
        if len(reg) == 2:
            return (float(reg[0]), float(reg[1]))
        raise ValueError(
            f'Regularizer expects 1 or 2 values (weight, bias), got {len(reg)}'
        )

    def _set_regularizers(self, l1=(), l2=()):
        """Helper for trainable layers to bind attributes required by Loss and Adam."""
        self.weight_regularizer_l1, self.bias_regularizer_l1 = self._parse_reg(l1)
        self.weight_regularizer_l2, self.bias_regularizer_l2 = self._parse_reg(l2)
        