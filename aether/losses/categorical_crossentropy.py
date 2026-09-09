import numpy as np
import aether.config as config
# Using Python 3.15 PEP 810: Explicit lazy imports would allow 
# for the import below to be called lazily, however for best 
# compatability this project would not use Python 3.15 for this issue
from aether.layers.activations import SoftMax
import aether.custom_kernels.loss_kernel as gpu_loss

def _to_sparse_labels(xp, y_true):
    """Canonicalize labels to sparse class-index form (S,).
    """
    return y_true if y_true.ndim == 1 else xp.argmax(y_true, axis=1)


def _cce_per_sample_loss(xp, probs_clip, y_true_sparse, n_classes, label_smoothing, training):
    """Per-sample CCE loss via direct gather"""
    samples = len(probs_clip)
    sample_idx = xp.arange(samples)
    target_probs = probs_clip[sample_idx, y_true_sparse]

    if label_smoothing > 0 and training:
        sum_log = xp.sum(xp.log(probs_clip), axis=1)
        return -(1.0 - label_smoothing) * xp.log(target_probs) - (label_smoothing / n_classes) * sum_log

    return -xp.log(target_probs)


class Loss:
    """Base interface and accumulator for objective functions and regularization penalties.

    Coordinates sample-loss accumulation, graph topology validation, and L1/L2
    weight decay across registered trainable layers.
    """
    prohibited_preceding_layers = ()  # Default: base loss forbids nothing
    has_fused_activation = False
    def __init__(self):
        self.new_pass()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def validate_graph(self, layers: list):
        """Validates that the model graph architecture is compatible with this loss function."""

        last_layer = layers[-1]
        if isinstance(last_layer, self.prohibited_preceding_layers):
            prohibited_names = ", ".join(cls.__name__ for cls in self.prohibited_preceding_layers)
            raise ValueError(
                f"[aether] '{type(self).__name__}' operates directly on unnormalized logits. "
                f"Do not add an explicit '{prohibited_names}' layer immediately preceding this loss."
            )

    def get_config(self) -> dict:
        """Override to return constructor kwargs required to reconstruct the loss.
        Used by Model.save() / Model.load()."""
        return {
            "label_smoothing": (
                float(self.label_smoothing) if self.label_smoothing is not None else None
            ),
        }
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization= False, training = True):
        xp = config.get_array_module(output)
        sample_losses = self.forward(output, y, training) #calc sample losses
        data_loss = xp.mean(sample_losses)      #calc mean/average losses

        self.accumulated_sum += xp.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization = False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss() 
    
    def regularization_loss(self):
        regularization_loss = 0             # if we don't do this, we risk overfitting.
                                            # We will have to denote partials for this too...
        for layer in getattr(self, "trainable_layers", []):        
            xp = config.get_array_module(layer.weights)
            
            weights_fp32 = layer.weights.astype(xp.float32, copy=False)
            biases_fp32 = layer.biases.astype(xp.float32, copy=False)

            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                        xp.sum(xp.abs(weights_fp32))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += 0.5 * layer.weight_regularizer_l2 * \
                                        xp.sum(weights_fp32 * weights_fp32)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                        xp.sum(xp.abs(biases_fp32))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += 0.5 * layer.bias_regularizer_l2 * \
                                        xp.sum(biases_fp32 * biases_fp32) 
        return regularization_loss


class CategoricalCrossEntropy(Loss):
    """Categorical cross-entropy loss for multiclass probability distributions.

    Evaluates the cross-entropy objective between target class indices and predicted
    probabilities, supporting optional label smoothing during training.

    Parameters
    ----------
    label_smoothing : float, default=0.0
        Smoothing factor in [0.0, 1.0). When greater than zero and in training mode,
        replaces hard target distributions with uniform smoothing over all classes.

    Notes
    -----
    - **Expected inputs:**
        - ``y_pred``: Probability distributions of shape ``(batch_size, n_classes)``.
          Must already be normalized (e.g., via preceding ``SoftMax`` layer).
        - ``y_true``: Target labels as sparse integer indices ``(batch_size,)`` or
          one-hot distributions ``(batch_size, n_classes)``.
    """
    def __init__(self, label_smoothing = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing 

    def forward(self, y_pred, y_true, training = True):
        xp = config.get_array_module(y_pred)
        y_pred_fp32 = y_pred.astype(xp.float32, copy=False)
        n_classes = y_pred.shape[1]
        y_pred_clip = xp.clip(y_pred_fp32, 1e-7, 1 - 1e-7) #.000001 -> .999999
        y_true_sparse = _to_sparse_labels(xp, y_true)

        return _cce_per_sample_loss(xp, y_pred_clip, y_true_sparse, n_classes, self.label_smoothing, training)

    def backward(self, logits, y_true, training=True):

        xp = config.get_array_module(logits)
        samples = len(logits)
        n_classes = logits.shape[1]

        logits_fp32 = logits.astype(xp.float32, copy=False)
        y_true_sparse = _to_sparse_labels(xp, y_true)
        logits_clip = xp.clip(logits_fp32, 1e-7, 1 - 1e-7)
        sample_idx = xp.arange(samples)

        dinputs = xp.zeros_like(logits)
        target_probs = logits_clip[sample_idx, y_true_sparse]

        if self.label_smoothing > 0 and training:
            dinputs += -(self.label_smoothing / n_classes) / logits_clip / samples
            dinputs[sample_idx, y_true_sparse] -= (1.0 - self.label_smoothing) / target_probs / samples
        else:
            dinputs[sample_idx, y_true_sparse] = -1.0 / target_probs / samples

        self.dinputs = dinputs.astype(logits.dtype, copy=False)
        return self.dinputs


class SoftmaxCategoricalCrossEntropy(Loss):
    """Fused Softmax activation and Categorical Cross-Entropy objective.

    Operates directly on unnormalized network logits, evaluating cross-entropy
    while collapsing the analytical gradient into a numerically stable difference
    `(probs - targets) / N`.

    Parameters
    ----------
    label_smoothing : float, default=0.0
        Smoothing factor in [0.0, 1.0). When greater than zero and in training mode,
        smooths the target ground truths prior to gradient calculation.

    Notes
    -----
    - **Expected inputs:**
        - ``logits``: Raw, unnormalized activations of shape ``(batch_size, n_classes)``.
        - ``y_true``: Target labels as sparse integer indices ``(batch_size,)`` or
          one-hot distributions ``(batch_size, n_classes)``.
    - **Graph constraint:** Operates directly on logits; preceding this loss with an
      explicit ``SoftMax`` layer is prohibited to prevent redundant double normalization.
    """
    prohibited_preceding_layers = (SoftMax,)
    has_fused_activation = True
    def __init__(self, label_smoothing = 0.0):
        super().__init__()
        self.activation = SoftMax()
        self.label_smoothing = label_smoothing

        self.backward = self._backward_fallback

    def _compile_for_device(self, device):
        """Triggered by Model.to(device) to bind the fused elementwise
        backward kernel or the fallback."""
        if device == 'cupy' and gpu_loss.is_gpu_softmax_cce_backward_available():
            self.backward = self._backward_gpu
        else:
            self.backward = self._backward_fallback

    #y_true is the vector of correct class indices, one per sample.
    #inputs is the raw logits, shape (n_samples, n_classes)
    def forward(self, logits, y_true, training = True):
        xp = config.get_array_module(logits)

        self.activation.forward(logits, training=training)  # call forward function of softmax
        probs = self.activation.output                

        probs_fp32 = probs.astype(xp.float32, copy=False)
        n_classes = probs_fp32.shape[1]
        y_true_sparse = _to_sparse_labels(xp, y_true)
        probs_clip = xp.clip(probs_fp32, 1e-7, 1 - 1e-7)

        # Per-sample array, NOT a reduced scalar -- Loss.calculate() means
        # to call xp.mean()/xp.sum()/len() on this directly.
        return _cce_per_sample_loss(xp, probs_clip, y_true_sparse, n_classes, self.label_smoothing, training)
    
    def predictions(self, outputs):
        """Mirrors the functionality of softmax predictions, we require this as we pop softmax in combined pass"""
        xp = config.get_array_module(outputs)
        return xp.argmax(outputs, axis = 1)

    def _backward_fallback(self, logits, y_true, training=True):
        xp = config.get_array_module(logits)
        samples = len(logits)
        n_classes = logits.shape[1]

        # Compute probabilities directly from the logits passed into backward
        probs = self.activation.forward(logits, training=training)
        probs_fp32 = probs.astype(xp.float32, copy=False)
        y_true_sparse = _to_sparse_labels(xp, y_true)
        sample_idx = xp.arange(samples)

        dinputs = probs_fp32.copy()
        if self.label_smoothing > 0 and training:
            dinputs -= self.label_smoothing / n_classes
            dinputs[sample_idx, y_true_sparse] -= (1.0 - self.label_smoothing)
        else:
            dinputs[sample_idx, y_true_sparse] -= 1.0

        dinputs /= samples
        self.dinputs = dinputs.astype(logits.dtype, copy=False)
        return self.dinputs

    def _backward_gpu(self, logits, y_true, training=True):
        xp = config.get_array_module(logits)
        samples = len(logits)
        n_classes = logits.shape[1]

        # Reuse probabilities computed during loss.forward()
        # (Avoids redundant SoftMax kernel execution)
        probs = self.activation.output
        probs_fp32 = probs.astype(xp.float32, copy=False)

        y_true_sparse = _to_sparse_labels(xp, y_true).astype(xp.int64, copy=False)
        class_idx = xp.arange(n_classes, dtype=xp.int64).reshape(1, n_classes)
        y_true_row = y_true_sparse.reshape(samples, 1)

        apply_smoothing = self.label_smoothing > 0 and training
        smooth_offset = np.float32(self.label_smoothing / n_classes if apply_smoothing else 0.0)
        target_offset = np.float32(1.0 - self.label_smoothing if apply_smoothing else 1.0)
        inv_samples = np.float32(1.0 / samples)

        dinputs = gpu_loss.softmax_cce_backward(
            probs_fp32, y_true_row, class_idx, smooth_offset, target_offset, inv_samples
        )

        self.dinputs = dinputs.astype(logits.dtype, copy=False)
        return self.dinputs