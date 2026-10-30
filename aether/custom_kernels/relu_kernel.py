import aether.config as config

_relu_backward_kernel = config.build_kernel(
    lambda: config.cp.ElementwiseKernel(
        'T dvalues, U output',
        'T dinputs',
        'dinputs = output > 0 ? dvalues : (T)0',
        'relu_backward_kernel'
    ),
    name='_relu_backward_kernel'
)

def is_gpu_relu_backward_available() -> bool:
    """Checks if CuPy hardware support and kernels are loaded."""
    return _relu_backward_kernel is not None

def relu_backward(dvalues, output):
    return _relu_backward_kernel(dvalues, output)
