from functools import partial
import numpy as np
import aether.config as config
from aether.base import Layer
from aether.custom_kernels import pooling_kernel as gpu_pooling
from aether.custom_kernels.launch_math import _compute_magic_numbers 

class _PoolNd(Layer):
    """Base class for MaxPool2d and AvgPool2d

    Handles shared logic between build, calculating padded input,
    and output shape calculation.
    """
    def __init__(self):
        super().__init__()
        self._launch_cache = {}

    def build(self, input_shape: tuple[int, ...], seed: int | None = None) -> tuple[int, ...]:
        """
        Computes spatial downsampled output for shape (H_out, W_out, C)
        for AvgPool2d and MaxPool2d, weights and seed are ignored
        """
        super().build(input_shape)

        H_in, W_in, C_in = input_shape
        fH, fW = self.filter_size
        sH, sW = self.stride

        if self.padding == "same":
            H_out = int(np.ceil(H_in / sH))
            W_out = int(np.ceil(W_in / sW))
        elif self.padding == "valid":
            H_out = int(np.floor((H_in - fH) / sH) + 1)
            W_out = int(np.floor((W_in - fW) / sW) + 1)
        else:
            raise ValueError(f"Unsupported padding mode '{self.padding}'. Use 'same' or 'valid'.")

        self.output_shape = (H_out, W_out, C_in)
        return self.output_shape

    def _get_padded_input(self, H_in, W_in):
        fH, fW = self.filter_size
        sH, sW = self.stride

        if self.padding == "valid":
            H_out = int(np.floor((H_in - fH) / sH + 1).item())
            W_out = int(np.floor((W_in - fW) / sW + 1).item())

            pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        elif self.padding == "same":

            H_out = int(np.ceil(H_in / sH).item())
            W_out = int(np.ceil(W_in / sW).item())

            pad_h = max((H_out - 1) * sH + fH - H_in, 0)
            pad_w = max((W_out - 1) * sW + fW - W_in, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
        else:
            raise ValueError(f"Expected padding == valid or same, recieved {self.padding} instead")
        return H_out, W_out, pad_top, pad_bottom, pad_left, pad_right
    
    def _prepare_forward_input(self, inputs, pad_value):
        """Shared shape validation + padding for pooling forward passes.
        pad_value differs by pooling type: -inf for max, 0.0 for average."""
        xp = config.get_array_module(inputs)
        if inputs.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, recieved {inputs.ndim}D tensor instead")

        inputs = inputs.astype(xp.float32, copy=False)
        S, H_in, W_in, C = inputs.shape
        H_out, W_out, pad_top, pad_bottom, pad_left, pad_right = self._get_padded_input(H_in, W_in)

        self.pad_top, self.pad_bottom = pad_top, pad_bottom
        self.pad_left, self.pad_right = pad_left, pad_right

        if self.padding == "valid":
            inputs_padded = inputs
        elif self.padding == "same":
            inputs_padded = xp.pad(
                inputs,
                ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=pad_value,
            )
        else:
            raise ValueError(f"Expected padding == valid or same, recieved {self.padding} instead")

        inputs_padded = xp.ascontiguousarray(inputs_padded)
        return xp, inputs_padded, S, H_out, W_out
    
    def _get_shape_meta(self, input_shape, block_z):
            """
            Based on deprecated _PoolNd._get_launch_geometry
            Retrieves or computes launch geometry, padding bounds, and scalar static arguments.
            Memoized per unique (input_shape, block_z) pair. On cache hits, executes in ~100ns.
            """
            cache_key = (input_shape, block_z)
            cached = self._launch_cache.get(cache_key)
            if cached is not None:
                return cached

            S, H_in, W_in, C = input_shape
            H_out, W_out, pad_top, pad_bottom, pad_left, pad_right = self._get_padded_input(H_in, W_in)

            H_pad = H_in + pad_top + pad_bottom
            W_pad = W_in + pad_left + pad_right

            fH, fW = self.filter_size
            sH, sW = self.stride

            # Block axes should combine to 1024 for CUDA (block_z=4)
            # or 512 for HIP (block_z=2)
            target_threads = 1024 if block_z == 4 else 512
            target_2d = target_threads // block_z # Always 256

            if C <= 32:
                block_x = max(1, C)
                block_y = target_2d // block_x
            else:
                block_x = min(32, C)
                block_y = 8

            block_dim = (block_x, block_y, block_z)

            grid_x = (C + block_x - 1) // block_x
            grid_y = (W_out + block_y - 1) // block_y
            grid_z = (H_out * S + block_z - 1) // block_z
            grid_dim = (grid_x, grid_y, grid_z)

            # Since GPU doesn't have dedicated GPU divsion units, perform the divsion 
            # algorithm needed to unsplit the grid_z axis on the CPU instead. 
            max_h_s = grid_z * block_z - 1
            magic_scale, magic_shift = _compute_magic_numbers(H_out, max_h_s)

            static_args = (
                np.int32(S), np.int32(H_pad), np.int32(W_pad), np.int32(C),
                np.int32(fH), np.int32(fW), np.int32(sH), np.int32(sW),
                np.int32(H_out), np.int32(W_out),
                magic_scale, magic_shift,
            )

            meta = {
                "S": S, "H_in": H_in, "W_in": W_in, "C": C,
                "H_out": H_out, "W_out": W_out,
                "H_pad": H_pad, "W_pad": W_pad,
                "padded_shape": (S, H_pad, W_pad, C),
                "out_shape": (S, H_out, W_out, C),
                "pad_top": pad_top, "pad_bottom": pad_bottom,
                "pad_left": pad_left, "pad_right": pad_right,
                "block_dim": block_dim,
                "grid_dim": grid_dim,
                "static_args": static_args,
            }
            self._launch_cache[cache_key] = meta
            return meta
    
    def _forward_gpu_common(self, inputs, pad_value, block_z, kernel, track_indices=False):
        """Shared GPU forward driver for 2D pooling layers.
        
        Handles shape extraction, padding, output (and optional index buffer) allocation,
        geometry memoization, and kernel dispatch.
        """
        xp, inputs_padded, S, H_out, W_out = self._prepare_forward_input(inputs, pad_value=pad_value)
        meta = self._get_shape_meta(inputs.shape, block_z)

        self.inputs_shape = inputs.shape
        self.padded_shape = meta["padded_shape"]

        self.output = xp.empty(meta["out_shape"], dtype=inputs_padded.dtype)

        if track_indices:
            self.max_indices = xp.empty(meta["out_shape"], dtype=xp.int32)
            aux_args = (self.max_indices,)
        else:
            self.max_indices = None
            aux_args = ()

        kernel_args = (inputs_padded, self.output) + aux_args + meta["static_args"]
        kernel(meta["grid_dim"], meta["block_dim"], kernel_args)

        return self.output

    def _backward_gpu_common(self, dvalues, block_z, kernel, aux_args=()):
        """Shared GPU backward driver using full shape memoization."""
        xp = config.get_array_module(dvalues)

        meta = self._get_shape_meta(self.inputs_shape, block_z)

        dvalues = xp.ascontiguousarray(dvalues.astype(xp.float32, copy=False))
        dinputs_padded = xp.zeros(meta["padded_shape"], dtype=dvalues.dtype)

        kernel_args = (dvalues,) + aux_args + (dinputs_padded,) + meta["static_args"]
        kernel(meta["grid_dim"], meta["block_dim"], kernel_args)

        self.dinputs = dinputs_padded[
            :,
            meta["pad_top"] : meta["pad_top"] + meta["H_in"],
            meta["pad_left"] : meta["pad_left"] + meta["W_in"],
            :,
        ]
        return self.dinputs

    def get_config(self):
        return {
            "filter_size":  self.filter_size,
            "stride":       self.stride,
            "padding": self.padding
        }
class MaxPool2d(_PoolNd):
    """2D Max Pooling layer for spatial downsampling over NHWC inputs.

    Reduces spatial dimensions (height, width) by sliding a window across the input
    and extracting the maximum activation within each region. The operation is applied
    independently per feature channel.

    Parameters
    ----------

    filter_size : tuple of int, default=(2, 2)
        Spatial dimensions of the pooling kernel as (height, width).
    stride : tuple of int, default=(2, 2)
        Stride step size along (height, width).
    padding : {"same", "valid"}, default="same"
        Padding mode. "same" pads the spatial borders so output dimensions scale
        proportionally by stride; "valid" applies no padding and drops boundary 
        pixels that do not fit a full window.

    Notes
    -----
    Expected input shape: ``(batch_size, in_height, in_width, channels)``        
    Output shape ``(batch_size, out_height, out_width, channels)``
    """
    def __init__(self, filter_size = (2, 2), stride = (2,2),
                 padding = 'valid'):

        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == 'cupy' and config.HAS_CUPY:
            variant, target_threads = config.resolve_gpu_launch_geometry()

            block_z = 2 if target_threads == 512 else 4 # for 1024 threads

            kernel_train = gpu_pooling.get_max_pool2d_forward_kernel(variant, training=True)
            kernel_infer = gpu_pooling.get_max_pool2d_forward_kernel(variant, training=False)
            kernel_backward = gpu_pooling.get_max_pool2d_backward_kernel(variant)

            if kernel_train is None or kernel_infer is None or kernel_backward is None:
                self.forward = self._forward_fallback
                self.backward = self._backward_fallback
                return

            self._variant = variant
            self._block_z = block_z
            self._kernel_train = kernel_train
            self._kernel_infer = kernel_infer
            self._kernel_backward_nonoverlap = kernel_backward

            self.forward = partial(self._forward_gpu, block_z=block_z)
    
            if self.filter_size == self.stride:
                self.backward = partial(self._backward_gpu, block_z=block_z)
            else:
                self.backward = self._backward_fallback
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training, block_z):
        kernel = self._kernel_train if training else self._kernel_infer
        return self._forward_gpu_common(
            inputs,
            pad_value=-np.inf,
            block_z=block_z,
            kernel=kernel,
            track_indices=training,
        )

    def _backward_gpu(self, dvalues, block_z):
        return self._backward_gpu_common(
            dvalues,
            block_z,
            kernel=self._kernel_backward_nonoverlap,
            aux_args=(self.max_indices,),
        )
    
    def _forward_fallback(self, inputs, training):

        xp, inputs_padded, S, H_out, W_out = self._prepare_forward_input(inputs, pad_value=-np.inf)
        as_strided = config.get_stride_utility(xp)

        # Save shapes for either backward path to un-pad dinputs
        self.inputs_shape = inputs.shape
        self.padded_shape = inputs_padded.shape

        _, H_pad, W_pad, C = inputs_padded.shape
        fH, fW = self.filter_size
        sH, sW = self.stride

        patches = as_strided(
            inputs_padded,
            shape = (S, H_out, W_out, fH, fW, C), 
            strides = (
                inputs_padded.strides[0],      # Step between samples
                inputs_padded.strides[1] * sH, # Step between rows
                inputs_padded.strides[2] * sW, # Step between columns
                inputs_padded.strides[1],      # Move down 1 row inside patch
                inputs_padded.strides[2],      # Move right 1 col inside patch
                inputs_padded.strides[3],      # Step between each channel
            )
        )
        pooled = patches.max(axis = (3, 4)) 
        flat_in_window = patches.reshape(S, H_out, W_out, fH * fW, C).argmax(axis=3)
        win_rows, win_cols = xp.unravel_index(flat_in_window, (fH, fW))

        # Absolute row/col in the *padded* input
        h_out_idx = xp.arange(H_out).reshape(1, H_out, 1, 1)
        w_out_idx = xp.arange(W_out).reshape(1, 1, W_out, 1)
        s_idx = xp.arange(S).reshape(S, 1, 1, 1)
        c_idx = xp.arange(C).reshape(1, 1, 1, C)

        h_in = h_out_idx * sH + win_rows
        w_in = w_out_idx * sW + win_cols

        # Same flat-index convention as the GPU kernel: ((s*H_pad + h_in)*W_pad + w_in)*C + c
        self.max_indices = (((s_idx * H_pad + h_in) * W_pad + w_in) * C + c_idx).astype(xp.int32)

        self.output = pooled
        return self.output

    def _backward_fallback(self, dvalues):
        xp = config.get_array_module(dvalues)

        S, H_pad, W_pad, C = self.padded_shape
        _, H_in, W_in, _ = self.inputs_shape
        flat_dinputs = xp.zeros(S * H_pad * W_pad * C, dtype=dvalues.dtype)
        xp.add.at(flat_dinputs, self.max_indices.ravel(), dvalues.ravel())
        dinputs_padded = flat_dinputs.reshape(S, H_pad, W_pad, C)

        self.dinputs = dinputs_padded[:, self.pad_top:self.pad_top + H_in,
                                        self.pad_left:self.pad_left + W_in, :]
        return self.dinputs

class AvgPool2d(_PoolNd):
    """2D Average Pooling layer for spatial downsampling over NHWC inputs.

    Reduces spatial dimensions (height, width) by sliding a window across the input
    and computing the mean activation within each region. The operation is applied
    independently per feature channel.

    Parameters
    ----------

    filter_size : tuple of int, default=(2, 2)
        Spatial dimensions of the pooling kernel as (height, width).
    stride : tuple of int, default=(2, 2)
        Stride step size along (height, width).
    padding : {"same", "valid"}, default="same"
        Padding mode. "same" pads the spatial borders so output dimensions scale
        proportionally by stride; "valid" applies no padding and drops boundaryboundary 
        pixels that do not fit a full window.

    Notes
    -----
    Expected input shape: ``(batch_size, in_height, in_width, channels)``        
    Output shape ``(batch_size, out_height, out_width, channels)``
    """
    def __init__(self, filter_size = (2, 2), stride = (2,2),
                 padding = 'valid'):

        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback


    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths."""
        if device == 'cupy' and config.HAS_CUPY:
            variant, target_threads = config.resolve_gpu_launch_geometry()
            block_z = 2 if target_threads == 512 else 4 # for 1024
            kernel_forward = gpu_pooling.get_avg_pool2d_forward_kernel(variant)
            kernel_backward = gpu_pooling.get_avg_pool2d_backward_kernel(variant)

            if kernel_forward is None or kernel_backward is None:
                self.forward = self._forward_fallback
                self.backward = self._backward_fallback
                return

            self._variant = variant
            self._block_z = block_z
            self._kernel_forward = kernel_forward
            self._kernel_backward_nonoverlap = kernel_backward

            self.forward = partial(self._forward_gpu, block_z=block_z)

            if self.filter_size == self.stride:
                self.backward = partial(self._backward_gpu, block_z=block_z)
            else:
                self.backward = self._backward_fallback
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training, block_z):
        return self._forward_gpu_common(
            inputs,
            pad_value=0.0,
            block_z=block_z,
            kernel=self._kernel_forward,
            track_indices=False,
        )

    def _backward_gpu(self, dvalues, block_z):
        return self._backward_gpu_common(
            dvalues,
            block_z,
            kernel=self._kernel_backward_nonoverlap,
            aux_args=(),
        )

    def _forward_fallback(self, inputs, training):

        xp, inputs_padded, S, H_out, W_out = self._prepare_forward_input(inputs, pad_value=0.0)
        as_strided = config.get_stride_utility(xp)

        # Save shapes for backward un-padding
        self.inputs_shape = inputs.shape
        self.padded_shape = inputs_padded.shape

        _, _, _, C = inputs_padded.shape
        fH, fW = self.filter_size
        sH, sW = self.stride

        patches = as_strided(
            inputs_padded,
            shape=(S, H_out, W_out, fH, fW, C), 
            strides=(
                inputs_padded.strides[0],      # Step between samples
                inputs_padded.strides[1] * sH, # Step between rows
                inputs_padded.strides[2] * sW, # Step between columns
                inputs_padded.strides[1],      # Move down 1 row inside patch
                inputs_padded.strides[2],      # Move right 1 col inside patch
                inputs_padded.strides[3],      # Step between each channel
            )
        )

        self.output = patches.mean(axis=(3, 4))
        return self.output

    def _backward_fallback(self, dvalues): 
            xp = config.get_array_module(dvalues)
            as_strided = config.get_stride_utility(xp)

            S, H_in, W_in, C = self.inputs_shape
            S, H_pad, W_pad, C = self.padded_shape
            _, H_out, W_out, _ = dvalues.shape
            fH, fW = self.filter_size
            sH, sW = self.stride

            # Dilate incoming gradients to account for strides
            dilated_H = (H_out - 1) * sH + 1
            dilated_W = (W_out - 1) * sW + 1

            dvalues_dilated = xp.zeros((S, dilated_H, dilated_W, C), dtype=dvalues.dtype)                
            dvalues_dilated[:, ::sH, ::sW, :] = dvalues
            
            backward_pad_top = fH - 1
            backward_pad_left = fW - 1
            backward_pad_bottom = H_pad - dilated_H
            backward_pad_right = W_pad - dilated_W

            dvalues_padded = xp.pad(
                dvalues_dilated, 
                pad_width=(
                    (0, 0), 
                    (backward_pad_top, backward_pad_bottom), 
                    (backward_pad_left, backward_pad_right), 
                    (0, 0)
                ),
                mode='constant',
                constant_values=0.0,
            )
            dvalues_patches = as_strided(
                dvalues_padded,
                shape=(S, H_pad, W_pad, fH, fW, C),
                strides=(
                    dvalues_padded.strides[0],       # step between samples
                    dvalues_padded.strides[1],       # step down a padded row
                    dvalues_padded.strides[2],       # step across a padded column 
                    dvalues_padded.strides[1],       # move down 1 row inside patch
                    dvalues_padded.strides[2],       # move right 1 col inside patch
                    dvalues_padded.strides[3],       # step across channels
                ),
            )

            # Average gradient across receptive windows
            dinputs_padded = dvalues_patches.sum(axis=(3, 4)) * (1.0 / (fH * fW))

            # Un-pad gradient back to original input shape
            self.dinputs = dinputs_padded[
                :, 
                self.pad_top : self.pad_top + H_in, 
                self.pad_left : self.pad_left + W_in, 
                :
            ]
            return self.dinputs

class GlobalAvgPool(Layer):
    """
    Global Average Pooling for NHWC inputs. 
    
    Reduces the feature channel to a single scaler by computing the
    mean over all spatial dimensions (height and width), collpasing 4D input 
    dimensions into 2D feature representations.

    Notes
    -----
    Expected input shape: ``(batch_size, height, width, channels)``        
    Output shape ``(batch_size, channels)``
    
    """
    def __init__(self):
        super().__init__()
        self._launch_cache = {}

    def build(self, input_shape: tuple[int, ...], seed: int | None = None) -> tuple[int, ...]:
        super().build(input_shape)
    
        C_in = input_shape[-1]
        
    
        self.output_shape = (C_in,)
        return self.output_shape
    
    def _get_shape_meta(self, input_shape):
        """
        Similar to _PoolNd._get_shape_meta, however a bit simpler meaning no need
        to refactor that method to include support for GAP. 
        """
        cache_key = (input_shape, self._target_threads)
        cached = self._launch_cache.get(cache_key)
        if cached is not None:
            return cached

        S, H, W, C = input_shape
        if S <= 0 or C <= 0 or H <= 0 or W <= 0:
            raise ValueError(f"GlobalAvgPool input dimensions must all be positive, got {input_shape}")

        inv_area = np.float32(1.0 / (H * W))
        block_x = min(32, C)
        max_y = self._target_threads // block_x
        block_y = min(S, max_y)
        block_dim = (block_x, block_y, 1)
        # Calculate grid geometry based on kernel mapping:
        # threadIdx.x / blockIdx.x -> Channel axis (C)
        # threadIdx.y / blockIdx.y -> Sample/Batch axis (S)
        grid_x = (C + block_x - 1) // block_x
        grid_y = (S + block_y - 1) // block_y
        grid_z = 1
        grid_dim = (grid_x, grid_y, grid_z)

        static_args = (
            np.int32(S), np.int32(H), np.int32(W), np.int32(C),
            np.float32(inv_area)
        )
        meta = {
            "S": S, "H": H, "W": W, "C": C,
            "out_shape": (S, C),
            "block_dim": block_dim,
            "grid_dim": grid_dim,
            "static_args": static_args,
        }
        self._launch_cache[cache_key] = meta
        return meta

    def _compile_for_device(self, device):

        if device == "cupy" and config.HAS_CUPY:
            variant, target_threads = config.resolve_gpu_launch_geometry()
            
            self.kernel_forward = gpu_pooling.get_gap_forward_kernel(variant)
            self.kernel_backward = gpu_pooling.get_gap_backward_kernel(variant)

            self._variant = variant
            self._target_threads = target_threads
            self.forward = self._forward_gpu
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training):
        xp = config.cp
        self.inputs_shape = inputs.shape
        meta = self._get_shape_meta(self.inputs_shape)

        # Allocate device array matching (S, C) output shape
        self.output = xp.empty(meta["out_shape"], dtype=inputs.dtype)
        kernel_args = (inputs, self.output) + meta["static_args"]
        
        self.kernel_forward(meta["grid_dim"], meta["block_dim"], kernel_args)
        return self.output

    def _backward_gpu(self, dvalues):
        xp = config.cp
        meta = self._get_shape_meta(self.inputs_shape)
        
        self.dinputs = xp.empty(self.inputs_shape, dtype=dvalues.dtype)
        kernel_args = (dvalues, self.dinputs) + meta["static_args"]
    
        self.kernel_backward(meta["grid_dim"], meta["block_dim"], kernel_args)
        return self.dinputs
    
    def _forward_fallback(self, inputs, training):
        xp = config.get_array_module(inputs)
        self.inputs_shape = inputs.shape
        self.output = xp.average(inputs, axis=(1, 2))
        return self.output

    def _backward_fallback(self, dvalues):
        xp = config.get_array_module(dvalues)
        S, H, W, C = self.inputs_shape
        avg_weight = np.float32(1.0 / (H * W))

        dvalues4d = dvalues[:, xp.newaxis, xp.newaxis, :]
        dvalues4d = xp.broadcast_to(dvalues4d, shape=(S,H,W,C))
        self.dinputs = dvalues4d * avg_weight
        return self.dinputs