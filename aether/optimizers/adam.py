import numpy as np
from functools import partial

import aether.config as config
from aether.custom_kernels import adam_kernel as gpu_adam

class Optimizer:
    """Base class for all Aether ML optimizers.

    Provides parameter tracking, regularized gradient calculations (L1/L2),
    and decoupled weight decay resolution.
    """

    def __init__(self, lr: float = 0.001, decay: float = 0.0):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.iterations = 0
        self.layers = []

    def init_params(self, trainable_layers: list):
        """Registers trainable layers passed from Model.finalize()."""
        self.layers = trainable_layers

    @staticmethod
    def _l1_subgradient(param, l1_lambda, xp):
        """Sub-gradient of L1 regularization matching the +1-at-zero convention."""
        return l1_lambda * xp.where(param < 0, -1.0, 1.0).astype(param.dtype)

    def _get_regularized_gradients(self, layer, xp):
        """Folds coupled L1/L2 regularizations configured on the layer into dweights/dbiases."""
        dweights = layer.dweights
        dbiases = getattr(layer, "dbiases", None)

        if getattr(layer, "weight_regularizer_l1", 0.0) > 0:
            dweights = dweights + self._l1_subgradient(
                layer.weights, layer.weight_regularizer_l1, xp
            )
        if getattr(layer, "weight_regularizer_l2", 0.0) > 0:
            dweights = dweights + 2.0 * layer.weight_regularizer_l2 * layer.weights

        if dbiases is not None:
            if getattr(layer, "bias_regularizer_l1", 0.0) > 0:
                dbiases = dbiases + self._l1_subgradient(
                    layer.biases, layer.bias_regularizer_l1, xp
                )
            if getattr(layer, "bias_regularizer_l2", 0.0) > 0:
                dbiases = dbiases + 2.0 * layer.bias_regularizer_l2 * layer.biases

        return dweights, dbiases

    def _resolve_weight_decay(self, layer) -> float:
        """Resolves decoupled weight decay coefficient (for AdamW-style optimizers)."""
        if getattr(layer, "no_weight_decay", False):
            return 0.0
        return getattr(self, "weight_decay", 0.0)

    def step(self):
        """Unified optimizer entry point executed once per training step."""
        raise NotImplementedError(
            f"Optimizer '{type(self).__name__}' must implement a step() method."
        )

    def get_config(self) -> dict:
        """Override to return constructor kwargs required to load optimzer.
        Used by Model.save() / Model.load()."""
        return {
            "lr": float(self.lr),
            "decay": float(self.decay),
            "epsilon": float(self.epsilon),
        }
    
# General starting learning rate for SGD is 1.0, with a decay down to 0.1. For Adam, a good starting 
# LR is 0.001 (1e-3), decaying down to 0.0001 (1e-4). Different problems may require different 
# values here, but these are decent to start.
class Adam(Optimizer):
    """Adaptive Moment Estimation (Adam) optimizer.

    Updates parameters using running estimates of the first moment (mean) and
    uncentered second moment (variance) of gradients, with bias correction applied
    to both estimates.

    Parameters
    ----------
    lr : float, default=0.001
        Initial learning rate (step size).
    decay : float, default=0.0
        Inverse-time learning rate decay factor:
        ``current_lr = lr / (1.0 + decay * iteration)``.
    epsilon : float, default=1e-7
        Small constant added to the denominator for numerical stability.
    beta_1 : float, default=0.9
        Exponential decay rate for first-moment momentum estimates.
    beta_2 : float, default=0.999
        Exponential decay rate for second-moment squared-gradient estimates.

    Notes
    -----
    Momentum buffers are maintained in float32 precision across all execution
    backends. Coupled L1/L2 penalties configured on registered layers are folded
    directly into the gradient updates during the step.
    """

    def __init__(self, lr=.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=.999):

        super().__init__(lr, decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2  # used to be known as our rho

        self._step_impl = self._step_fallback
        # Bound in _compile_for_device: the memoized fused RawKernel and its
        # vendor variant ('cuda'/'hip'). None on CPU/NumPy or if compilation failed.
        self._adamw_kernel = None
        self._variant = None

    def init_params(self, trainable_layers):
        """
        Called once during Model.finalize() to register trainable layers
        and pre-allocate optimizer and momentum cache buffers in fp32.
        """
        super().init_params(trainable_layers)
        xp = config.xp

        for layer in self.layers:
            layer.weight_momentums = xp.zeros_like(layer.weights, dtype=xp.float32)
            layer.weight_cache = xp.zeros_like(layer.weights, dtype=xp.float32)

            if getattr(layer, 'biases', None) is not None:
                layer.bias_momentums = xp.zeros_like(layer.biases, dtype=xp.float32)
                layer.bias_cache = xp.zeros_like(layer.biases, dtype=xp.float32)

    def _compile_for_device(self, device):
        """
        Triggered by Model.to(device) to bind the fused single-kernel RawKernel
        GPU path or the fallback. 
        """
        if device == 'cupy':
            variant, block_size = config.resolve_gpu_launch_geometry()
            kernel = gpu_adam._get_compiled_adamw_kernel(variant)
            if kernel is not None:
                self._variant = variant
                self._adamw_kernel = kernel
                self._step_impl = partial(self._step_gpu, block_size=block_size)
                return

        self._adamw_kernel = None
        self._variant = None
        self._step_impl = self._step_fallback

    def step(self):
        """Unified optimizer entry point executed once per training step"""
        if self.decay:
            self.current_lr = np.float32(
                self.lr * (1.0 / (1.0 + self.decay * self.iterations))
            )
        t = self.iterations + 1
        bias_correction_1 = np.float32(1.0 - (self.beta_1 ** t))
        bias_correction_2 = np.float32(1.0 - (self.beta_2 ** t))

        self._step_impl(bias_correction_1, bias_correction_2)

        self.iterations += 1

    @staticmethod
    def _l1_subgradient(param, l1_lambda, xp):
        """Sub-gradient of L1 regularization matching the +1-at-zero convention.
        Used only by the CPU/NumPy fallback path -- the GPU path folds L1/L2
        directly into the fused kernel instead (see _step_gpu / adam_kernel.py).
        """
        return l1_lambda * xp.where(param < 0, -1.0, 1.0).astype(param.dtype)

    def _get_regularized_gradients(self, layer, xp):
        """
        Folds any *coupled* L1/L2 regularization configured on the
        layer itself (weight_regularizer_l1/l2, bias_regularizer_l1/l2)
        into dweights/dbiases. CPU fallback path only -- see note above.
        """
        dweights = layer.dweights
        dbiases = layer.dbiases

        if layer.weight_regularizer_l1 > 0:
            dweights = dweights + self._l1_subgradient(layer.weights, layer.weight_regularizer_l1, xp)
        if layer.weight_regularizer_l2 > 0:
            dweights = dweights + layer.weight_regularizer_l2 * layer.weights

        if layer.bias_regularizer_l1 > 0:
            dbiases = dbiases + self._l1_subgradient(layer.biases, layer.bias_regularizer_l1, xp)
        if layer.bias_regularizer_l2 > 0:
            dbiases = dbiases + layer.bias_regularizer_l2 * layer.biases

        return dweights, dbiases

    def _resolve_weight_decay(self, layer):
        """Resolves decoupled weight decay coefficient (AdamW).

        Layers such as BatchNorm set no_weight_decay=True since gamma/beta
        are 1D scale/shift parameters -- shrinking them toward zero degrades
        the normalization math rather than regularizing anything.
        """
        if getattr(layer, "no_weight_decay", False):
            return 0.0
        return getattr(self, "weight_decay", 0.0)

    def _step_gpu(self, bias_correction_1, bias_correction_2, block_size):
        """
        Fused single-kernel-per-tensor RawKernel path. L1/L2 regularization
        and decoupled weight decay are handled in the same kernel launches here
        """
        kernel = self._adamw_kernel

        lr = np.float32(self.current_lr)
        beta1 = np.float32(self.beta_1)
        beta2 = np.float32(self.beta_2)
        eps = np.float32(self.epsilon)
        bc1 = np.float32(bias_correction_1)
        bc2 = np.float32(bias_correction_2)

        for layer in self.layers:
            weight_decay = np.float32(self._resolve_weight_decay(layer))

            # Weights (or BatchNorm gamma, aliased to layer.weights)
            gpu_adam.launch_adamw_update(
                kernel,
                layer.weights, layer.dweights,
                layer.weight_momentums, layer.weight_cache,
                lr, beta1, beta2, eps, bc1, bc2,
                weight_decay,
                getattr(layer, "weight_regularizer_l1", 0.0),
                getattr(layer, "weight_regularizer_l2", 0.0),
                block_size=block_size
            )

            # Biases (or BatchNorm beta, aliased to layer.biases) --
            # weight decay is always 0.0 here, independent of no_weight_decay.
            if getattr(layer, 'biases', None) is not None and getattr(layer, 'dbiases', None) is not None:
                gpu_adam.launch_adamw_update(
                    kernel,
                    layer.biases, layer.dbiases,
                    layer.bias_momentums, layer.bias_cache,
                    lr, beta1, beta2, eps, bc1, bc2,
                    np.float32(0.0),
                    getattr(layer, "bias_regularizer_l1", 0.0),
                    getattr(layer, "bias_regularizer_l2", 0.0),
                    block_size=block_size
                )

            # Invalidate any low-precision compute casts stored on the layer
            if hasattr(layer, "invalidate_shadow_caches"):
                layer.invalidate_shadow_caches()

    def _step_fallback(self, bias_correction_1, bias_correction_2):
        """CPU / NumPy vectorized update path."""
        xp = config.xp
        learning_rate = np.float32(self.current_lr)
        epsilon = np.float32(self.epsilon)
        beta_1 = np.float32(self.beta_1)
        beta_2 = np.float32(self.beta_2)
        one_minus_beta_1 = np.float32(1.0) - beta_1
        one_minus_beta_2 = np.float32(1.0) - beta_2

        for layer in self.layers:
            dweights, dbiases = self._get_regularized_gradients(layer, xp)

            # Decoupled weight decay (AdamW)
            weight_decay = np.float32(self._resolve_weight_decay(layer))
            if weight_decay > 0.0:
                layer.weights -= learning_rate * weight_decay * layer.weights

            # Update weight momentums and second moment cache
            layer.weight_momentums = beta_1 * layer.weight_momentums + one_minus_beta_1 * dweights
            layer.weight_cache = beta_2 * layer.weight_cache + one_minus_beta_2 * (dweights ** 2)

            weight_momentums_corrected = layer.weight_momentums / bias_correction_1
            weight_cache_corrected = layer.weight_cache / bias_correction_2

            layer.weights -= learning_rate * weight_momentums_corrected / (
                xp.sqrt(weight_cache_corrected) + epsilon
            )

            # Update bias momentums and cache (if biases are present)
            if dbiases is not None:
                layer.bias_momentums = beta_1 * layer.bias_momentums + one_minus_beta_1 * dbiases
                layer.bias_cache = beta_2 * layer.bias_cache + one_minus_beta_2 * (dbiases ** 2)

                bias_momentums_corrected = layer.bias_momentums / bias_correction_1
                bias_cache_corrected = layer.bias_cache / bias_correction_2

                layer.biases -= learning_rate * bias_momentums_corrected / (
                    xp.sqrt(bias_cache_corrected) + epsilon
                )

            # Invalidate any low-precision compute casts stored on the layer
            if hasattr(layer, "invalidate_shadow_caches"):
                layer.invalidate_shadow_caches()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
        })
        return config
    
class AdamW(Adam):
    """Adam optimizer with decoupled weight decay (AdamW).

    Applies weight decay directly to parameters rather than folding it into the
    gradient updates alongside momentum and variance tracking.

    Parameters
    ----------
    lr : float, default=0.001
        Initial learning rate (step size).
    decay : float, default=0.0
        Inverse-time learning rate decay factor.
    epsilon : float, default=1e-7
        Small constant added to the denominator for numerical stability.
    beta_1 : float, default=0.9
        Exponential decay rate for first-moment momentum estimates.
    beta_2 : float, default=0.999
        Exponential decay rate for second-moment squared-gradient estimates.
    weight_decay : float, default=0.01
        Decoupled weight decay coefficient applied directly to weights:
        ``weights -= lr * weight_decay * weights``.

    Notes
    -----
    Layers marking ``no_weight_decay = True`` (e.g., ``BatchNorm`` scaling and
    shift vectors) and layer bias parameters are automatically exempt from decay.
    """
    def __init__(self, lr=.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=.999, weight_decay=0.01):
        
        super().__init__(lr=lr, decay=decay, epsilon=epsilon,
                          beta_1=beta_1, beta_2=beta_2)

        self.weight_decay = weight_decay

    def get_config(self) -> dict:
        config = super().get_config()
        config["weight_decay"] = float(self.weight_decay)
        return config