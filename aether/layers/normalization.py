import numpy as np
import aether.config as config
from aether.base import Layer
import aether.custom_kernels.batchnorm_kernel as gpu_bn

class BatchNorm(Layer):
    """Batch normalization layer over trailing feature channels.

    Normalizes inputs across all leading batch and spatial dimensions such that
    each feature channel along the final axis has zero mean and unit variance,
    followed by learnable affine scaling (`gamma`) and shifting (`beta`).

    Parameters
    ----------
    epsilon : float, default=1e-5
        Small constant added to batch variance to prevent division by zero.
    momentum : float, default=0.9
        Exponential moving average decay factor for tracking running statistics:
        ``running_stat = momentum * running_stat + (1 - momentum) * batch_stat``.

    Notes
    -----
    - **Expected layout:** Channels-last. Operates over 2D inputs of shape
      ``(batch_size, n_features)`` or 4D spatial inputs of shape
      ``(batch_size, height, width, n_channels)``.
    - **Shape contract:**
        - Input: ``(d_0, d_1, ..., d_{k-1}, C)``
        - Output: ``(d_0, d_1, ..., d_{k-1}, C)`` (identical to input)
    - **Training vs. Evaluation:**
        - In training mode (``training=True``), mean and variance are computed
          from the current mini-batch across all leading axes, and running
          statistics are updated.
        - In inference mode (``training=False``), inputs are normalized using the
          accumulated ``running_mean`` and ``running_var``.
    """
    
    _precision_exempt: bool = True
    no_weight_decay: bool = True  # decaying towards zero breaks normalization math

    def __init__(self, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self._launch_cache = {}

        self.n_features = None
        self.gamma = None
        self.beta = None
        self.weights = self.gamma
        self.biases = self.beta
        self.running_mean = None
        self.running_var = None

        # backward cache
        self.inputs = None
        self.normalized = None
        self.inv_std = None
        self.batch_mean = None
        self.batch_var = None

        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0

        self._variant = None
        self._mean_kernel = None
        self._var_kernel = None
        self._grad_sums_kernel = None

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback
        
    def _get_shape_meta(self, input_shape: tuple[int, ...]):
        """
        Retrieves or computes launch geometry and scalar static arguments.
        Memoized per unique input_shape. On cache hits, executes in ~100ns.
        """
        cache_key = input_shape
        cached = self._launch_cache.get(cache_key)
        if cached is not None:
            return cached

        C = self.n_features
        N = int(np.prod(input_shape)) // C
        block_y = gpu_bn.BLOCK_Y

        block_dim = (32, block_y, 1)
        grid_x = (C + 31) // 32
        grid_y = min(64, (N + block_y - 1) // block_y)
        grid_dim = (grid_x, grid_y, 1)

        static_args = (
            np.int32(N),
            np.int32(C),
        )

        meta = {
            "N": N,
            "C": C,
            "block_dim": block_dim,
            "grid_dim": grid_dim,
            "static_args": static_args,
        }
        self._launch_cache[cache_key] = meta
        return meta

    def build(self, input_shape: tuple[int, ...], seed: int | None = None) -> tuple[int, ...]:
        super().build(input_shape)

        if self.n_features is None:
            self.n_features = input_shape[-1]

        dim = int(self.n_features)
        xp = config.xp

        self.gamma = xp.ones(dim, dtype=xp.float32)
        self.beta = xp.zeros(dim, dtype=xp.float32)
        self.running_mean = xp.zeros(dim, dtype=xp.float32)
        self.running_var = xp.ones(dim, dtype=xp.float32)

        self.weights = self.gamma
        self.biases = self.beta

        return self.output_shape

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == "cupy" and config.HAS_CUPY and gpu_bn.is_gpu_bn_available():
            variant, _ = config.resolve_gpu_launch_geometry()

            mean_kernel = gpu_bn.get_bn_mean_kernel(variant)
            var_kernel = gpu_bn.get_bn_var_kernel(variant)
            grad_sums_kernel = gpu_bn.get_bn_grad_sums_kernel(variant)

            if (
                mean_kernel is None
                or var_kernel is None
                or grad_sums_kernel is None
                or gpu_bn._bn_forward_ew is None
                or gpu_bn._bn_backward_ew is None
            ):
                self.forward = self._forward_fallback
                self.backward = self._backward_fallback
                return

            self._variant = variant
            self._mean_kernel = mean_kernel
            self._var_kernel = var_kernel
            self._grad_sums_kernel = grad_sums_kernel

            self.forward = self._forward_gpu
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training):
        xp = config.cp
        if not inputs.flags.c_contiguous:
            inputs = xp.ascontiguousarray(inputs)

        if not training:
            inv_std = 1.0 / xp.sqrt(self.running_var + self.epsilon)
            scale = self.gamma * inv_std
            bias = self.beta - self.running_mean * scale

            output = xp.empty_like(inputs)
            gpu_bn._bn_inference_ew(inputs, scale, bias, output)
            return output

        meta = self._get_shape_meta(inputs.shape)
        N = meta["N"]
        C = meta["C"]

        inputs_flat = inputs.reshape(N, C)
        out_mean = xp.zeros((C,), dtype=xp.float32)

        kernel_args = (
            inputs_flat,
            None,
            None,
            out_mean,
            None,
        ) + meta["static_args"]
        self._mean_kernel(meta["grid_dim"], meta["block_dim"], kernel_args)

        batch_mean = out_mean / float(N)

        out_var = xp.zeros((C,), dtype=xp.float32)
        kernel_args = (
            inputs_flat,
            None,
            batch_mean,
            out_var,
            None,
        ) + meta["static_args"]
        self._var_kernel(meta["grid_dim"], meta["block_dim"], kernel_args)

        batch_var = out_var / float(N)
        inv_std = 1.0 / xp.sqrt(batch_var + self.epsilon)

        # Update running stats
        self.running_mean = (
            self.momentum * self.running_mean + (1.0 - self.momentum) * batch_mean
        )
        self.running_var = (
            self.momentum * self.running_var + (1.0 - self.momentum) * batch_var
        )

        # Cache intermediate state
        self.inputs = inputs
        self.batch_mean = batch_mean
        self.batch_var = batch_var
        self.inv_std = inv_std
        self.normalized = (inputs - batch_mean) * inv_std

        output = xp.empty_like(inputs)
        gpu_bn._bn_forward_ew(inputs, batch_mean, inv_std, self.gamma, self.beta, output)
        return output

    def _backward_gpu(self, dvalues):
        xp = config.xp
        if not dvalues.flags.c_contiguous:
            dvalues = xp.ascontiguousarray(dvalues)
        if not self.inputs.flags.c_contiguous:
            self.inputs = xp.ascontiguousarray(self.inputs)

        meta = self._get_shape_meta(self.inputs.shape)
        N = meta["N"]
        C = meta["C"]

        dvalues_flat = dvalues.reshape(N, C)
        inputs_flat = self.inputs.reshape(N, C)

        dbeta = xp.zeros((C,), dtype=xp.float32)
        diff_sum = xp.zeros((C,), dtype=xp.float32)

        kernel_args = (
            dvalues_flat,
            inputs_flat,
            self.batch_mean,
            dbeta,
            diff_sum,
        ) + meta["static_args"]
        self._grad_sums_kernel(meta["grid_dim"], meta["block_dim"], kernel_args)

        inv_std = self.inv_std
        gamma = self.gamma

        dgamma = inv_std * diff_sum
        dvar = gamma * diff_sum * (-0.5) * (inv_std**3)
        dmu = -inv_std * gamma * dbeta

        self.dbiases = dbeta
        self.dweights = dgamma

        gamma_inv_std = gamma * inv_std
        dvar_scaled = dvar * (2.0 / N)
        dmu_over_n = dmu / float(N)

        self.dinputs = xp.empty_like(dvalues)
        gpu_bn._bn_backward_ew(
            dvalues,
            self.inputs,
            self.batch_mean,
            gamma_inv_std,
            dvar_scaled,
            dmu_over_n,
            self.dinputs,
        )
        return self.dinputs

    def _forward_fallback(self, inputs, training):
        xp = config.get_array_module(inputs)

        if not training:
            inv_std = 1.0 / xp.sqrt(self.running_var + self.epsilon)
            scale = self.gamma * inv_std
            bias = self.beta - self.running_mean * scale
            return inputs * scale + bias

        axis = tuple(range(inputs.ndim - 1))

        batch_mean = xp.mean(inputs, axis=axis, keepdims=True)
        batch_var = xp.var(inputs, axis=axis, keepdims=True)

        # Update running stats
        self.running_mean = (
            self.momentum * self.running_mean + (1.0 - self.momentum) * batch_mean.squeeze()
        )
        self.running_var = (
            self.momentum * self.running_var + (1.0 - self.momentum) * batch_var.squeeze()
        )

        # Cache intermediate state
        self.inputs = inputs
        self.batch_mean = batch_mean
        self.batch_var = batch_var
        self.inv_std = 1.0 / xp.sqrt(batch_var + self.epsilon)
        self.normalized = (inputs - batch_mean) * self.inv_std

        return self.gamma * self.normalized + self.beta

    def _backward_fallback(self, dvalues):
        xp = config.get_array_module(dvalues)
        axes = tuple(range(dvalues.ndim - 1))
        N_total = self.inputs.size // self.inputs.shape[-1]

        # Gradients with respect to gamma and beta
        self.dweights = xp.sum(dvalues * self.normalized, axis=axes)
        self.dbiases = xp.sum(dvalues, axis=axes)

        dhatx = dvalues * self.gamma

        dvar = xp.sum(
            dhatx * (self.inputs - self.batch_mean) * (-0.5) * (self.inv_std**3),
            axis=axes,
            keepdims=True,
        )
        N_total_f = float(N_total)

        dmu = xp.sum(dhatx * -self.inv_std, axis=axes, keepdims=True) + dvar * xp.sum(
            -2.0 * (self.inputs - self.batch_mean), axis=axes, keepdims=True
        ) / N_total_f

        self.dinputs = (
            dhatx * self.inv_std
            + dvar * 2.0 * (self.inputs - self.batch_mean) / N_total_f
            + dmu / N_total_f
        )
        return self.dinputs

    def get_config(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "momentum": self.momentum,
        }

    def get_parameters(self) -> dict:
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    def set_parameters(
        self,
        gamma=None,
        beta=None,
        running_mean=None,
        running_var=None,
        **kwargs,
    ):
        if gamma is not None:
            self.gamma = gamma
            self.weights = self.gamma

        if beta is not None:
            self.beta = beta
            self.biases = self.beta

        if running_mean is not None:
            self.running_mean = running_mean

        if running_var is not None:
            self.running_var = running_var