import numpy as np
import aether.config as config
from aether.base import Layer

class ReLU(Layer):

    def forward(self, inputs, training):
        xp = config.get_array_module(inputs)
        self.output = xp.maximum(0, inputs)
        return self.output
    
    def backward(self, dvalues): 
        self.dinputs = dvalues * (self.output > 0)
        return self.dinputs

@config.fuse_kernel()
def _fused_leaky_relu_forward(x, alpha):
    return config.cp.maximum(0, x) + alpha * config.cp.minimum(0, x)

@config.fuse_kernel()
def _fused_leaky_relu_backward(dvalues, output, alpha):
    # ROCm/HIP Workaround: Comparing fusion variables against raw Python literals 
    # (e.g., output > 0) causes a type-guessing bug in modern NumPy environments, 
    # throwing an AttributeError on AMD architectures during the AST trace.
    # Generating a zero scalar explicitly typed by multiplying 'alpha * 0'
    # ensures variable-to-variable comparison, bypassing weak type-promotion logic 
    # and compiling cleanly into a branchless SIMD loop on both CUDA and HIP stacks.
    fused_zero = alpha * 0
    result = dvalues * (1.0 - (output <= fused_zero) * (1.0 - alpha))
    return result.astype(dvalues.dtype)

class LeakyReLU(Layer):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.precision_policy = config.DTypePolicy(compute_dtype=None)
        self._alpha_compute = self.precision_policy.cast_to_compute(np.float32(self.alpha))

        self.forward = self._forward_fallback
        self.backward = self._backward_fallback

    def _apply_precision(self, policy):
        """Called on Model.set_precision(); re-derives alpha in the active compute dtype."""
        self.precision_policy = policy or config.DTypePolicy()
        self._alpha_compute = self.precision_policy.cast_to_compute(np.float32(self.alpha))

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to map low-level hardware paths"""
        if device == 'cupy' and _fused_leaky_relu_forward is not None and _fused_leaky_relu_backward is not None:
            self.forward = self._forward_gpu
            self.backward = self._backward_gpu
        else:
            self.forward = self._forward_fallback
            self.backward = self._backward_fallback

    def _forward_gpu(self, inputs, training):
        self.output = _fused_leaky_relu_forward(inputs, self._alpha_compute)
        return self.output

    def _backward_gpu(self, dvalues):
        self.dinputs = _fused_leaky_relu_backward(dvalues, self.output, self._alpha_compute)
        return self.dinputs

    def _forward_fallback(self, inputs, training):
        xp = config.get_array_module(inputs)
        self.output = xp.maximum(0, inputs) + self.alpha * xp.minimum(0, inputs)
        return self.output

    def _backward_fallback(self, dvalues):
        self.dinputs = dvalues * (1.0 - (self.output <= 0) * (1.0 - self.alpha))
        return self.dinputs

    def get_config(self):
        return {
            "alpha": self.alpha,
        }

class SoftMax(Layer):
    _precision_exempt = True 
    def forward(self, inputs, training=False):
        xp = config.get_array_module(inputs)

        if inputs.dtype != xp.float32:
            inputs = inputs.astype(xp.float32, copy=False)

        exp_values = xp.exp(inputs - xp.max(inputs, axis=1, keepdims = True)) #e**(inputs - max(inputs by row))
        probabilities = exp_values / xp.sum(exp_values, axis=1, keepdims = True) #e**k / sum(e**k) 
        self.output = probabilities

        return self.output

    # A vectorized pass of the SoftMax backwards pass
    def backward(self, dvalues): 
        xp = config.get_array_module(dvalues)

        if dvalues.dtype != xp.float32:
            dvalues = dvalues.astype(xp.float32, copy=False)
            
        sum_dvalues_output = xp.sum(dvalues * self.output, axis = -1, keepdims=True)
        self.dinputs = self.output * (dvalues - sum_dvalues_output)

    def predictions(self, outputs):
        xp = config.get_array_module(outputs)
        return xp.argmax(outputs, axis = 1) #return the max of the rows