import warnings
import numpy as np
import aether.config as config
from aether.base import Layer
from aether.custom_kernels import conv_kernel
class Conv2d(Layer):
    """2D Convolutional layer for spatial feature extraction over NHWC inputs.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor (e.g., 3 for RGB).
    out_channels : int, default=1
        Number of filters / feature maps to produce.
    filter_size : tuple of int, default=(3, 3)
        Spatial dimensions of the convolutional kernel as (height, width).
    stride : tuple of int, default=(1, 1)
        Stride step size along (height, width).
    padding : {"same", "valid"}, default="same"
        Padding mode. "same" preserves spatial dimensions (with stride 1),
        "valid" applies no padding.
    l1 : float or tuple, default=()
        L1 penalty. Pass a float for weights only, or (weight, bias).
    l2 : float or tuple, default=()
        L2 penalty. Pass a float for weights only, or (weight, bias).
    
    Warns
    -----
    UserWarning
        If the configured ``precision_policy`` requests a compute precision
        that does not match the active execution path (e.g., requesting a dtype
        other than ``float16`` when bound to the WMMA matrix-core path, or
        requesting non-``float32`` when falling back to strided CPU/GPU kernels).
    
    Notes
    -----
    Expected input shape: ``(batch_size, height, width, in_channels)``
    Output shape: ``(batch_size, out_height, out_width, out_channels)``

    Conv2d's compute dtype is bounded to fp16 as WMMA kernels can't take >fp16
    calculations, instead layer will use the fallback vectorized path.
    """
    
    def __init__(self, in_channels, out_channels = 1, filter_size = (3, 3), stride = (1, 1), padding = "same", l1=(), l2=()):
        super().__init__()
        # input_shape has form (batch_size, height, width, channels)
        self.C_in = in_channels
        self.C_out = out_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding 

        self._set_regularizers(l1, l2)

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback
        self._fp16_weight_cache = None
        self._fp16_weight_valid = False
        self._launch_cache = {}
        # None until _compile_for_device runs. self.forward can't stand in for
        # this: the CPU branch there binds the same fallback __init__ does, so
        # it can't tell "never compiled" from "compiled for CPU".
        self._compiled_device = None
        self._precision_warned_key = None

        self.weights = None
        self.biases = None

    def build(self, input_shape: tuple[int, ...], seed: int | None = None) -> tuple[int, ...]:
        """
        Calculates output spatial dimensions, dynamically extracts C_in,
        and initializes filter weights with He normal initialization.
        """
        super().build(input_shape)
        xp = config.xp
        H_in, W_in, incoming_C_in = input_shape
        if self.C_in != incoming_C_in:
            raise ValueError(
                f"[aether] Shape mismatch in Conv layer: configured with C_in={self.C_in}, "
                f"but received input shape with dimension {incoming_C_in} ({input_shape})."
            )
        
        fH, fW = self.filter_size
        sH, sW = self.stride

        # 1. Compute Output Spatial Dimensions
        if self.padding == "same":
            H_out = int(np.ceil(H_in / sH))
            W_out = int(np.ceil(W_in / sW))
        elif self.padding == "valid":
            H_out = int(np.floor((H_in - fH) / sH) + 1)
            W_out = int(np.floor((W_in - fW) / sW) + 1)
        else:
            raise ValueError(f"Unsupported padding mode '{self.padding}'. Use 'same' or 'valid'.")

        self.output_shape = (H_out, W_out, self.C_out)

        fan_in = fH * fW * self.C_in
        std = (2.0 / fan_in) ** 0.5

        weight_shape = (fH, fW, self.C_in, self.C_out)
        effective_seed = seed if seed is not None else self.seed

        if effective_seed is not None:
            self.seed = effective_seed
            rng = xp.random.RandomState(effective_seed)
            raw_weights = rng.randn(*weight_shape)
        else:
            raw_weights = xp.random.randn(*weight_shape)

        self.filter_weights = (raw_weights * std).astype(xp.float32, copy=False)
        self.weights = self.filter_weights
        
        self.biases = xp.zeros(self.C_out, dtype=xp.float32)

        return self.output_shape
    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to bind the matrix-core path when available."""
        if device == 'cupy' and conv_kernel.get_is_conv_gpu_available():
            self.forward = self._forward_gpu
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback
        # Device may have underlying weight buffers 0 force a shadow refresh
        self._fp16_weight_valid = False
        self._compiled_device = device
        # Covers set_precision() -> to(); the binding is only knowable here.
        self._warn_if_precision_ignored()

    def _apply_precision(self, policy):
        """
        Called on Model.set_precision(). Conv2d's kernels pin their own operand
        dtype, so the policy is stored for introspection/serialization only --
        _warn_if_precision_ignored is what keeps that from being silent.
        """
        self.precision_policy = policy or config.DTypePolicy()
        # Before _compile_for_device runs there is no binding to check against,
        # so defer to it rather than warning about a placeholder.
        if self._compiled_device is not None:
            self._warn_if_precision_ignored()

    def _effective_compute_dtype(self):
        """The dtype this layer actually multiplies in, read off the live binding."""
        # Compare __func__: attribute access builds a fresh bound method each
        # time, so `self.forward is self._forward_gpu` is never True.
        is_gpu = getattr(self.forward, '__func__', None) is Conv2d._forward_gpu
        return 'float16' if is_gpu else 'float32'

    def _warn_if_precision_ignored(self):
        """
        Warn once per (bound path, requested dtype) when the precision policy
        asks for something this layer's kernels can't deliver. Fires at bind
        time only -- never from forward/backward.
        """
        requested = (
            self.precision_policy.compute_dtype_name if self.precision_policy else None
        )
        if requested is None:
            return

        effective = self._effective_compute_dtype()
        if requested == effective:
            self._precision_warned_key = None
            return

        # Absorbs the to() + finalize() double-compile while still letting a
        # genuinely different mismatch warn again.
        key = (effective, requested)
        if key == self._precision_warned_key:
            return
        self._precision_warned_key = key

        path = (
            "the matrix-core path, whose WMMA kernels take float16 operands at "
            "the hardware level (float32 accumulate, float32 output)"
            if effective == 'float16' else
            "the strided fallback path, which computes in float32"
        )
        warnings.warn(
            f"[aether] Conv2d: precision policy requests {requested!r}, but this "
            f"layer is bound to {path}. The policy is ignored for this layer.",
            UserWarning,
        )

    def _refresh_fp16_weights(self, xp):
        """
        Shadow fp16 cast of filter weights, update optimizer kernels
        and classes to cast these back to f32, and back to f16
        """

        fH, fW, C_in, C_out = self.filter_weights.shape
        if self._fp16_weight_cache is None:
            self._fp16_weight_cache = xp.empty((fH * fW * C_in, C_out), dtype=xp.float16)
        if not self._fp16_weight_valid:
            self._fp16_weight_cache[...] = self.filter_weights.reshape(fH * fW * C_in, C_out).astype(xp.float16)
            self._fp16_weight_valid = True
    
    def invalidate_shadow_caches(self):
        """
        Called by an optimizer after it writes into self.weights'
        underlying buffer in place (e.g. a fused GPU kernel that takes
        the array as an output pointer) -- that kind of update never
        goes through the filter_weights/weights property setter, so
        _fp16_weight_valid would otherwise stay True and _forward_gpu
        would keep matrix-multiplying against a stale fp16 snapshot.
        Forces a rebuild on the next call that needs it.
        """
        self._fp16_weight_valid = False

    def _get_shape_meta(self, spatial_shape):
        """Cached ConvShapeMeta for a spatial input shape (batch-independent)."""
        cache_key = (spatial_shape, self.filter_size, self.stride, self.padding)

        cached = self._launch_cache.get(cache_key)
        if cached is not None:
            return cached

        shape_meta = conv_kernel.ConvShapeMeta.create(
            spatial_shape, self.C_out, self.filter_size, self.stride, self.padding
        )
        self._launch_cache[cache_key] = shape_meta
        return shape_meta

    def _forward_gpu(self, inputs, training):
        xp = config.get_array_module(inputs)
        S, H_in, W_in, C_in = inputs.shape

        shape_meta = self._get_shape_meta((H_in, W_in, C_in))
        plan = conv_kernel.plan_forward_launch(shape_meta, S)

        # Fall back if JIT failed or shape isn't supported by Matrix Cores
        if plan is None:
            self.forward = self._forward_fallback
            return self._forward_fallback(inputs, training)

        inputs_fp16 = inputs.astype(xp.float16, copy=False)

        if not self._fp16_weight_valid:
            self._refresh_fp16_weights(xp)

        self.output = xp.empty(
            (S, shape_meta.H_out, shape_meta.W_out, self.C_out), dtype=xp.float32
        )

        plan(inputs_fp16, self._fp16_weight_cache, self.biases, self.output)

        self.inputs = inputs
        return self.output

    def _backward_gpu(self, dvalues):
        xp = config.xp
        S, H_in, W_in, C_in = self.inputs.shape
        fH, fW = self.filter_size

        shape_meta = self._get_shape_meta((H_in, W_in, C_in))
        dw_plan = conv_kernel.plan_dweight_launch(shape_meta, S)
        dx_plan = conv_kernel.plan_dinput_launch(shape_meta, S)

        if dw_plan is None or dx_plan is None:
            raise RuntimeError(
                f"Failed to compile GPU backward kernels for Conv shape "
                f"(H_in={H_in}, W_in={W_in}, C_in={C_in}, C_out={self.C_out})."
            )

        dvalues_fp32 = dvalues.astype(xp.float32, copy=False)
        dvalues_bounded = xp.clip(dvalues_fp32, -65504.0, 65504.0)
        dvalues_fp16 = dvalues_bounded.astype(xp.float16, copy=False)
        inputs_fp16 = self.inputs.astype(xp.float16, copy=False)

        if not self._fp16_weight_valid:
            self._refresh_fp16_weights(xp)

        self.dbiases = xp.zeros((self.C_out,), dtype=xp.float32)
        self.dweights = xp.zeros((fH, fW, C_in, self.C_out), dtype=xp.float32)
        self.dinputs = xp.empty_like(self.inputs, dtype=xp.float32)

        dw_plan(inputs_fp16, dvalues_fp16, self.dweights, self.dbiases)
        dx_plan(dvalues_fp16, self._fp16_weight_cache, self.dinputs)

        return self.dinputs

    def _forward_fallback(self, inputs, training):
        
        xp = config.get_array_module(inputs)
        as_strided = config.get_stride_utility(xp)
        fH, fW = self.filter_size
        sH, sW = self.stride
        S, H_in, W_in, D_in = inputs.shape

        pad_h = (fH - 1) // 2 if self.padding == "same" else 0
        pad_w = (fW - 1) // 2 if self.padding == "same" else 0

        # Retrieve spatial geometry - keyed on stride/pad too, since both
        # are mutable instance attrs and this cache is per-instance.
        spatial_key = (H_in, W_in, D_in, sH, sW, pad_h, pad_w, "fallback")
        meta = self._launch_cache.get(spatial_key)

        if meta is None:
            H_out = int((H_in + 2 * pad_h - fH) / sH + 1)
            W_out = int((W_in + 2 * pad_w - fW) / sW + 1)

            meta = {
                "pad_h": pad_h,
                "pad_w": pad_w,
                "H_out": H_out,
                "W_out": W_out,
            }
            self._launch_cache[spatial_key] = meta

        pad_h, pad_w = meta["pad_h"], meta["pad_w"]
        H_out, W_out = meta["H_out"], meta["W_out"] 
        # (0, 0) -> don't touch the number of samples in the batch
        # (P, P) -> pad top and bottom pixels by P pixels (axis 1)
        # (P, P) -> pad left and right pixels by P pixels (axis 2)
        # (0, 0) -> don't pad depth. 
        # contstant -> add constant_values for the padded values
        padded_inputs = xp.pad(array = inputs, 
                            pad_width = ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                            mode = 'constant',
                            constant_values = 0).astype(xp.float32, copy = False)

        self.output = xp.zeros((S, H_out, W_out, self.C_out), dtype = xp.float32)
        self.patches_shape=(S, H_out, W_out, fH, fW, D_in)
        self.patches_strides = (
                padded_inputs.strides[0],       # step between samples
                padded_inputs.strides[1] * sH,  # step down a row
                padded_inputs.strides[2] * sW,  # step across a column
                padded_inputs.strides[1],       # move down 1 row inside patch
                padded_inputs.strides[2],       # move right 1 col inside patch
                padded_inputs.strides[3],       # step across channels
        )

        patches = as_strided(
            padded_inputs,
            shape=self.patches_shape,
            strides=self.patches_strides)

        # Keep the samples, H_out, W_out, and C_out. 
        # But, iterate over the patch(x, y) with channels c, and with the number of filters d
        self.output = xp.einsum('shwxyc,xycd->shwd', patches, self.filter_weights, optimize=True)
        self.output += self.biases.reshape((1, 1, 1, self.C_out)) 

        self.inputs = inputs
        self.padded_inputs = padded_inputs
        return self.output

    
    def _backward_fallback(self, dvalues):

        xp = config.get_array_module(dvalues)
        as_strided = config.get_stride_utility(xp)

        S, H_out, W_out, C_out = dvalues.shape
        fH, fW, C_in, C_out = self.filter_weights.shape
        sH, sW = self.stride
        _, H_in, W_in, _ = self.inputs.shape 

        patches = as_strided(
            self.padded_inputs,
            shape=self.patches_shape,
            strides=self.patches_strides
        )
        # Now we need to account for dbiases and dweights
        self.dbiases = xp.sum(dvalues, axis = (0, 1, 2))
        self.dweights = xp.tensordot(patches, dvalues, axes = ([0, 1, 2], [0, 1, 2]))

        dilated_H = (H_out - 1) * sH + 1
        dilated_W = (W_out - 1) * sW + 1 
        
        dvalues_dilated = xp.zeros(shape= (S, dilated_H, dilated_W, C_out), dtype= dvalues.dtype)
        # Inject values using step slices
        dvalues_dilated[:, ::sH, ::sW, :] = dvalues
        
        pad_h = (fH - 1) // 2 if self.padding == "same" else 0
        pad_w = (fW - 1) // 2 if self.padding == "same" else 0

        pad_top  = (fH - 1) - pad_h
        pad_left = (fW - 1) - pad_w
        pad_bottom = H_in + fH - 1 - pad_top - dilated_H
        pad_right  = W_in + fW - 1 - pad_left - dilated_W

        dvalues_padded = xp.pad(dvalues_dilated, pad_width= (
            (0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            )

        windows_h = dvalues_padded.shape[1] - fH + 1
        windows_w = dvalues_padded.shape[2] - fW + 1
        if windows_h != H_in or windows_w != W_in:
            raise RuntimeError(
                f"[aether] Conv2d backward: dilated dY padding yields "
                f"{windows_h}x{windows_w} windows for a {H_in}x{W_in} input "
                f"(filter={fH}x{fW}, stride={sH}x{sW}, padding='{self.padding}'). "
                f"Refusing to build an out-of-bounds strided view."
            )

        flipped_weights = self.filter_weights[::-1, ::-1, :, :]
        dvalues_patches = as_strided(dvalues_padded, 
                            shape = (S, H_in, W_in, fH, fW, C_out),
                            strides=(
                                dvalues_padded.strides[0],  # Batch step
                                dvalues_padded.strides[1],  # Window grid row step (backward stride = 1)
                                dvalues_padded.strides[2],  # Window grid col step (backward stride = 1)
                                dvalues_padded.strides[1],  # Internel window pixel row step
                                dvalues_padded.strides[2],  # Internel window pixel col step
                                dvalues_padded.strides[3]   # Output channel step
                            ))
    
        self.dinputs = xp.tensordot(
            dvalues_patches, 
            flipped_weights, 
            axes=([3, 4, 5], [0, 1, 3]) # Match fH, fW, C_out
        )

        return self.dinputs

    def get_config(self) -> dict:
        return {
            "in_channels": self.C_in,
            "out_channels": self.C_out,
            "filter_size": self.filter_size,
            "stride": self.stride,
            "padding": self.padding,
            "l1": (self.weight_regularizer_l1, self.bias_regularizer_l1),
            "l2": (self.weight_regularizer_l2, self.bias_regularizer_l2),
        }

    def get_parameters(self) -> dict:
        return {
            "weights": self.weights,    # accesses self.filter_weights under the hood
            "biases": self.biases,
        }

    def set_parameters(self, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
            self._fp16_weight_valid = False  # Invalidate stale runtime cache on parameter load
        if biases is not None:
            self.biases = biases

    @property
    def filter_weights(self):
        return self._filter_weights

    @filter_weights.setter
    def filter_weights(self, value):
        # Automatically reshape 2D weight matrices to 4D filter tensors
        if value is not None and value.ndim == 2:
            fH, fW = self.filter_size
            value = value.reshape(fH, fW, self.C_in, self.C_out)
        self._filter_weights = value
        self._fp16_weight_valid = False

    @property
    def weights(self):
        return self.filter_weights

    @weights.setter
    def weights(self, value):
        self.filter_weights = value