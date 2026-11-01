import numpy as np

import aether.config as config
from aether.base import Layer
from aether.custom_kernels import dropout_kernel as gpu_dropout


class _DropoutBase(Layer):
    """
    Shared state and RNG plumbing for stochastic dropout layers.
    """
    is_stochastic = True
    def __init__(self, rate, seed=None):
        super().__init__()

        if not 0.0 <= rate < 1.0:
            raise ValueError(
                f"[aether] Dropout rate must be in the interval [0.0, 1.0), got {rate}."
            )

        self.rate = float(rate)
        self.keep_rate = 1.0 - self.rate
        self._inv_keep = 1.0 / self.keep_rate
        # The GPU kernel compares 16-bit draws, so its keep rate is threshold / 65536.
        # Scaling by the inverse of that rate keeps the expectation exact.
        self._keep_threshold = max(1, round(self.keep_rate * 65536))
        self._gpu_inv_keep = 65536 / self._keep_threshold

        self.seed = seed
        self._seed_key = config.derive_stream_seed(seed, 0)

        self._clock = config.TrainingClock()
        self._active_offset = -1

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback

    def _bind_rng(self, *, base_seed, stream_id, clock):
        """
        Called once by Model.finalize(). An explicit per-layer seed takes precedence
        over the model seed; the graph position (stream_id) is always mixed in to ensure
        layers never share a mask stream. The clock is rebound unconditionally so the
        model owns step advancement.
        """
        base = self.seed if self.seed is not None else base_seed
        self._seed_key = config.derive_stream_seed(base, stream_id)
        self._clock = clock

    def _make_generator(self, xp, offset):
        if xp is np:
            return np.random.Generator(
                np.random.Philox(key=self._seed_key, counter=offset)
            )
        return xp.random.default_rng(config.derive_stream_seed(self._seed_key, offset))

    def _require_offset(self):
        if self._active_offset < 0:
            raise RuntimeError(
                f"[aether] {type(self).__name__}.backward() called without a preceding "
                "training forward pass. The dropout mask cannot be reconstructed."
            )
        return self._active_offset

    @staticmethod
    def _apply_mask(x, mask):
        """A dropped element is exactly 0: multiplying alone would turn NaN or inf into NaN."""
        xp = config.get_array_module(x)
        return xp.where(mask != 0, x, x.dtype.type(0.0)) * mask

    def get_config(self):
        return {
            "rate": self.rate,
            "seed": self.seed,
        }


class Dropout(_DropoutBase):
    """Applies inverted dropout to inputs during training.
    
    Randomly zeros individual elements with probability `1- keep_rate`
    and scales retained elements by `1 / keep_rate` to preserve expectation.
    
    Notes
    -----
    Identity is passed during inference, skipping and dropout calculation.
    On GPU the rate rounds to the nearest 1/65536 (error at most 7.6e-6), so rates
    below 7.6e-6 drop nothing. No mask is stored: backward rebuilds it from the seed.
    """
    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == "cupy" and gpu_dropout.is_gpu_dropout_available():
            self.forward = self._forward_gpu
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training):

        if not training:
            self._active_offset = -1
            return inputs

        offset = self._clock.value
        self._active_offset = offset

        output = gpu_dropout.philox_dropout(
            inputs, self._seed_key, offset, self._keep_threshold, self._gpu_inv_keep
        )
        if output is None:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback
            return self._forward_fallback(inputs, training)

        self.output = output
        return self.output

    def _backward_gpu(self, dvalues):

        dinputs = gpu_dropout.philox_dropout(
            dvalues, self._seed_key, self._require_offset(), self._keep_threshold, self._gpu_inv_keep
        )
        if dinputs is None:
            raise RuntimeError(
                f"[aether] Dropout: failed to compile the GPU kernel for {dvalues.dtype} gradients."
            )

        self.dinputs = dinputs
        return self.dinputs

    def _forward_fallback(self, inputs, training):

        if not training:
            self._active_offset = -1
            return inputs
        offset = self._clock.value
        self._active_offset = offset

        xp = config.get_array_module(inputs)
        gen = self._make_generator(xp, offset)

        dtype = inputs.dtype
        keep = dtype.type(self._inv_keep)
        drop = dtype.type(0.0)

        uniform = gen.random(inputs.shape, dtype=np.float32)
        self.binary_mask = xp.where(uniform < self.keep_rate, keep, drop)

        self.output = self._apply_mask(inputs, self.binary_mask)
        return self.output

    def _backward_fallback(self, dvalues):
        self._require_offset()
        self.dinputs = self._apply_mask(dvalues, self.binary_mask)
        return self.dinputs


class SpatialDropout(_DropoutBase):
    """Channel-wise dropout for NHWC feature maps (Tompson et al., 2015).

    Drops entire 2D feature channels along the trailing axis with probability
    `1 - keep_rate` instead of individual pixels. This promotes independence
    across channels. Each sample draws its own channel mask, as in Keras
    `SpatialDropout2D` and PyTorch `Dropout2d`.

    Notes
    -----
    Identity is passed during inference, skipping and dropout calculation.
    On GPU the rate rounds to the nearest 1/65536 (error at most 7.6e-6), so rates
    below 7.6e-6 drop nothing. The (S, 1, 1, C) mask stays cached for backward.
    """

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == "cupy" and gpu_dropout.is_gpu_dropout_available():
            self.forward = self._forward_gpu
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training):

        if not training:
            self._active_offset = -1
            self.channel_mask = None
            return inputs

        offset = self._clock.value
        self._active_offset = offset

        xp = config.get_array_module(inputs)
        ones = xp.ones((inputs.shape[0], 1, 1, inputs.shape[-1]), dtype=xp.float32)
        mask = gpu_dropout.philox_dropout(
            ones, self._seed_key, offset, self._keep_threshold, self._gpu_inv_keep
        )
        if mask is None:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback
            return self._forward_fallback(inputs, training)

        self.channel_mask = mask.astype(inputs.dtype, copy=False)
        self.output = gpu_dropout.apply_channel_mask(inputs, self.channel_mask)
        return self.output

    def _backward_gpu(self, dvalues):
        self._require_offset()
        self.dinputs = gpu_dropout.apply_channel_mask(dvalues, self.channel_mask)
        return self.dinputs

    def _forward_fallback(self, inputs, training):

        if not training:
            self._active_offset = -1
            return inputs

        offset = self._clock.value
        self._active_offset = offset

        xp = config.get_array_module(inputs)
        gen = self._make_generator(xp, offset)

        self.C = inputs.shape[-1]
        dtype = inputs.dtype
        keep = dtype.type(self._inv_keep)
        drop = dtype.type(0.0)

        uniform = gen.random((inputs.shape[0], 1, 1, self.C), dtype=np.float32)
        self.channel_mask = xp.where(uniform < self.keep_rate, keep, drop)

        self.output = self._apply_mask(inputs, self.channel_mask)
        return self.output

    def _backward_fallback(self, dvalues):
        self._require_offset()
        self.dinputs = self._apply_mask(dvalues, self.channel_mask)
        return self.dinputs