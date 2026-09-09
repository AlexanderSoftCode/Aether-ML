import json
import warnings
import zipfile
from functools import partial
from pathlib import Path
import numpy as np

import aether.config as config
import aether.layers as layer_module
import aether.losses as loss_module
import aether.metrics as metric_module
import aether.optimizers as optimizer_module
import aether.preprocessing as transforms_module

from aether.base import Layer
from aether._utils import NullAccuracy, NullOptimizer, NullPreprocessor
from aether._utils.progress import make_progress
try:
    import safetensors.numpy
except ImportError:
    safetensors = None

# Manifest schema. A major bump means semantics an older reader cannot honour
# and is a hard error on load; a minor bump means additive keys an older reader
# simply ignores, so it only warns.
_SCHEMA_MAJOR = 1
_SCHEMA_MINOR = 1
_SCHEMA_VERSION = f"{_SCHEMA_MAJOR}.{_SCHEMA_MINOR}"

# Preprocessors are absent: their configs nest further {class_name, config}
# entries, so they resolve through aether.preprocessing.deserialize instead.
_COMPONENT_NAMESPACES = {
    "loss": (loss_module, loss_module.Loss),
    "optimizer": (optimizer_module, optimizer_module.Optimizer),
    "accuracy": (metric_module, metric_module.Accuracy),
}

def _serialize_component(obj):
    if obj is None or isinstance(obj, (NullOptimizer, NullAccuracy, NullPreprocessor)):
        return None
    return {
        "class_name": type(obj).__name__,
        "config": obj.get_config(),
    }


def _resolve_component(entry, kind):
    """Reconstructs a training component from a {class_name, config} manifest entry."""
    if not entry:
        return None

    class_name = entry["class_name"]
    cfg = entry.get("config") or {}

    namespace, base_cls = _COMPONENT_NAMESPACES[kind]
    component_cls = getattr(namespace, class_name, None)

    # Unknown class check
    if component_cls is None:
        raise ValueError(
            f"[aether] Unknown {kind} class '{class_name}' found in saved manifest. "
            f"Only classes defined in '{namespace.__name__}' can be deserialized."
        )

    # Base class rejection
    if component_cls is base_cls or not (isinstance(component_cls, type) and issubclass(component_cls, base_cls)):
        raise TypeError(
            f"[aether] Manifest entry '{class_name}' is not a valid concrete "
            f"'{base_cls.__name__}' subclass."
        )

    try:
        return component_cls(**cfg)
    except TypeError as exc:
        raise TypeError(
            f"[aether] Could not reconstruct {kind} '{class_name}' from saved config "
            f"{cfg}. The class signature may have changed since the model was saved."
        ) from exc

class Model():
    def __init__(self):
        self.layers = []
        self.is_finalized = False
        self.loss = None
        self.optimizer = None
        self.accuracy = None
        self.preprocessor = None
        self._seed = None
        self._predict_activation = None
        self._rng_clock = None

    def add(self, layer):
        """Appends a layer to the execution graph."""
        if self.is_finalized:   
            raise RuntimeError("Cannot modify model after finalize() has been called.")
        
        if not isinstance(layer, Layer):
            raise TypeError(
                f"Expected an instance of 'Layer', but got `{type(layer).__name__}`."
                "Make sure the layer you are passing in inherits from aether.base.Layer"
            )
        self.layers.append(layer)

    def manual_seed(self, seed: int):
        if self.is_finalized:
            raise RuntimeError("Cannot set a new seed after finalize() has been called.")
        self._seed = int(seed)
    
    def configure(self, loss=None, optimizer=None, accuracy=None, preprocessor=None):
        """Configures the training components (loss, optimizer, accuracy metric, preprocessor)
        for the model.

        Each argument, if provided, is validated against its expected base class.
        """
        if loss is None and optimizer is None and accuracy is None and preprocessor is None:
            raise ValueError(
                "At least one component (loss, optimizer, metrics, or preprocessor)"
                " must be provided to configure()."
            )

        if loss is not None:
            if not isinstance(loss, loss_module.Loss):
                raise TypeError(
                    f"Expected an instance of 'Loss', but got {type(loss).__name__}."
                    "Make sure the loss your are passing in inherits from aether.losses.loss"
                )
            else:
                self.loss = loss

        if optimizer is not None:
            if not isinstance(optimizer, optimizer_module.Optimizer):
                raise TypeError(
                    f"Expected an instance of 'Optimizer', but got '{type(optimizer).__name__}'. "
                    "Make sure the optimizer you are passing in inherits from aether.optimizers.Optimizer."
                )
            else:
                self.optimizer = optimizer

        if accuracy is not None:
            if not isinstance(accuracy, metric_module.Accuracy):
                raise TypeError(
                    f"Object '{type(accuracy).__name__}' is not a valid accuracy metric."
                    "Make sure the accuracy your passing in inherits from aether.metrics.accuracy"
                )
            self.accuracy = accuracy

        if preprocessor is not None:
            if not isinstance(preprocessor, transforms_module.Preprocess):
                raise TypeError(
                    f"Expected an instance of 'Preprocess', but got '{type(preprocessor).__name__}'. "
                    "Make sure the preprocessor you are passing in inherits from "
                    "aether.preprocessing.Preprocess."
                )
            self.preprocessor = preprocessor

    def _sync_device(self, target_device=None):
        """Internal dispatch that compiles all registered components for the active backend."""
        dev = target_device or getattr(self, "device", None)
        if dev is None:
            return

        # Consolidate all model items (layers + training modules)
        components = [
            *self.layers,
            getattr(self, "loss", None),
            getattr(self, "optimizer", None),
            getattr(self, "accuracy", None),
            getattr(self, "preprocessor", None),
        ]

        for comp in components:
            if comp is not None and hasattr(comp, "_compile_for_device"):
                comp._compile_for_device(dev)

    def to(self, device):
        """
        Ahead-of-time compilation and device migration.

        Configures the global execution backend (NumPy or CuPy) and recursively
        migrates every registered component to it, compiling dedicated kernels
        when targeting CuPy. Only trainable layers hold tensors, so those are
        the only ones updated in place -- meaning this can be called again
        after training.

        Args:
            device (str): The target hardware execution device, either 'cupy' or 'numpy'.
        Raises:
            ValueError: If `device` specifies an unsupported or unconfigured backend.
        Example:
            >>> model = ae.Model()
            ... model.add(...)
            ... model.add(...)
            ... model.to("cupy")
            ... # Now we can train and evaluate on the CuPy backend
            ... model.configure(...)
            ... model.finalize(input_shape=(...))
            ... model.train(X, y, epochs=5, shuffle=True)
            ... model.evaluate(X_val, y_val)
            ... model.to("numpy")
            ... # We can perform inference on the CPU now
            ... preds = model.predict(X_cpu)
        """
        target_device = device.lower()
        config.set_backend(target_device)
        self.device = target_device
        self._sync_device(target_device=target_device)

        if self.is_finalized:
            for layer in self.trainable_layers:
                param_dict = layer.get_parameters()
                migrated_dict = {}

                for name, tensor in param_dict.items():
                    if tensor is not None:
                        migrated_dict[name] = config.to_device(tensor, target=target_device)

                layer.set_parameters(**migrated_dict)

    def set_precision(self, compute_dtype):
        """
        Sets the target floating-point precision policy on the model and dispatches
        it to every registered layer, skipping any layer marked with the
        `_precision_exempt` attribute, then to the attached preprocessor, which is
        skipped under the same flag. 

        Args:
            compute_dtype (str): The target floating-point precision -- one of
                'float16', 'float32', or 'float64'.
        Raises:
            TypeError: If `compute_dtype` is not a str or None.
            ValueError: If `compute_dtype` is not one of the currently supported precisions.
            RuntimeError: If `compute_dtype` is not supported by NumPy.
        Note:
            This method can be called before or after both `Model.to()` and
            `Model.finalize()`, but must be called before training or inference.
        """
        self.precision_policy = config.DTypePolicy(compute_dtype)
        for layer in self.layers:
            if (hasattr(layer, "_apply_precision")
                 and not getattr(layer, "_precision_exempt", False)):
                layer._apply_precision(self.precision_policy)

        if self.preprocessor is not None and not getattr(
            self.preprocessor, "_precision_exempt", False
        ):
            self.preprocessor._apply_precision(self.precision_policy)

    def finalize(self, input_shape: tuple[int, ...]):
        """
        Finalizes the model architecture, propagating tensor dimensions
        and allocating weights/buffers across all registered layers.
        """
        if self.is_finalized:
            raise RuntimeError("Cannot modify model after finalize() has been called.")

        if not self.layers:
            raise RuntimeError(
                "[aether] Cannot finalize an empty model. Please add layers via Model.add() first."
            )

        # Reuse a clock restored by load(); otherwise start a fresh stream at step 0.
        if getattr(self, "_rng_clock", None) is None:
            self._rng_clock = config.TrainingClock()

        current_shape = input_shape
        for idx, layer in enumerate(self.layers):
            layer_seed = (self._seed + idx) if self._seed is not None else None
            # Unconditionally build and propagate shape
            try:
                current_shape = layer.build(current_shape, seed=layer_seed)
            except TypeError:
                current_shape = layer.build(current_shape)

            # Stochastic layers derive a per-stream key from the model seed and their
            # position in the graph, and share the model-wide step counter.
            if layer.is_stochastic:
                layer._bind_rng(
                    base_seed=self._seed,
                    stream_id=idx,
                    clock=self._rng_clock,
                )

        self._stochastic_layers = [layer for layer in self.layers if layer.is_stochastic]

        self.trainable_layers = [
            layer for layer in self.layers
            if getattr(layer, "weights", None) is not None or getattr(layer, "biases", None) is not None
        ]

        if self.loss is not None:
            self.loss.validate_graph(layers=self.layers)
            self.loss.remember_trainable_layers(self.trainable_layers)

            if self.loss.has_fused_activation:
                self._predict_activation = self.loss.activation

        if self.optimizer is not None:
            if not hasattr(self.optimizer, "step") or not hasattr(self.optimizer, "init_params"):
                raise TypeError(
                    f"Optimizer '{type(self.optimizer).__name__}' must implement "
                    "'init_params(trainable_layers)' and 'step()'."
                )
            self.optimizer.init_params(self.trainable_layers)
        else:
            self.optimizer = NullOptimizer()

        self._step_optimizer = self.optimizer.step

        if self.accuracy is None:
            self.accuracy = NullAccuracy()

        if self.preprocessor is None:
            self.preprocessor = NullPreprocessor()

        precision_policy = getattr(self, "precision_policy", None)
        if precision_policy is not None and not getattr(
            self.preprocessor, "_precision_exempt", False
        ):
            self.preprocessor._apply_precision(precision_policy)

        if getattr(self, "device", None) is None:
            self.device = "cupy" if getattr(config.xp, "__name__") == "cupy" else "numpy"

        self._sync_device()
        self.is_finalized = True

    def _check_ready(self, op_name, need_fitted=False):
        """Guard shared by train/evaluate/predict: the model must be finalized,
        and -- for evaluate/predict, which never fit on the caller's behalf --
        the attached preprocessor must already be fitted.
        """
        if not self.is_finalized:
            raise RuntimeError(
                f"[aether] Model must be explicitly finalized before {op_name}. "
                "Call model.finalize(input_shape) first."
            )
        if need_fitted and not self.preprocessor.is_fitted:
            raise RuntimeError(
                f"[aether] The attached preprocessing pipeline is not fitted, so {op_name} "
                "would run on untransformed data. Fit the pipeline before attaching it "
                "(e.g. Compose([...]).fit(X_train)), or train the model first -- "
                "model.train() fits an unfitted pipeline automatically."
            )

    def _has_pipeline(self):
        return self.preprocessor is not None and not isinstance(self.preprocessor, NullPreprocessor)

    def _assert_pipeline_device(self, X):
        """Probe the attached pipeline once and confirm it emits on the model's device.

        Transforms a single-sample slice X[:1] to verify pipeline device matches models
        expected device (NumPy or CuPy) before full execution begins.

        Args:
            X (ndarray): The raw input array whose first sample is used as the probe.
        Raises:
            TypeError: If the pipeline's output does not live on the model's device.
        """
        expected_backend = getattr(self, "device", "numpy")
        probe = self.preprocessor.transform(X[:1])
        actual_backend = "cupy" if type(probe).__module__.startswith("cupy") else "numpy"

        if actual_backend != expected_backend:
            raise TypeError(
                f"[aether] Preprocessing pipeline device mismatch: model is configured for "
                f"the '{expected_backend}' backend, but the attached "
                f"'{type(self.preprocessor).__name__}' produced a '{type(probe).__name__}' "
                f"on '{actual_backend}'. Add an aether.ToTensor(...) transform to the pipeline "
                f"(model.to() retargets it automatically), or migrate the model via "
                f"model.to('{actual_backend}')."
            )

    def _make_target_preparer(self, y):
        """Select the per-batch label device before training.

        Slicing preserves an array's module, so whether `y` needs to move is fully
        knowable before the loop. Labels are migrated per batch rather than once
        up front so that `train()`'s shuffled fancy-indexing keeps operating on `y`
        in its original namespace, matching the index array drawn from `X`.

        Args:
            y (ndarray): The full target array passed to train()/evaluate().
        Returns:
            callable: Identity when `y` already matches the model's device,
            otherwise a `config.to_device` partial bound to that device.
        """
        expected_backend = getattr(self, "device", "numpy")
        is_y_cupy = type(y).__module__.startswith("cupy")

        if (expected_backend == "cupy") == is_y_cupy:
            return lambda batch_y: batch_y
        return partial(config.to_device, target=expected_backend)

    def _assert_device_alignment(self, X, y=None):
        """Ensure incoming input tensors strictly match the configured hardware backend."""
        expected_backend = getattr(self, "device", "numpy")
        
        is_x_cupy = type(X).__module__.startswith("cupy")
        is_y_cupy = type(y).__module__.startswith("cupy") if y is not None else is_x_cupy

        if expected_backend == "cupy":
            if not is_x_cupy or not is_y_cupy:
                raise TypeError(
                    f"[aether] Device mismatch: Model is configured for CuPy backend ('cupy'), "
                    f"but received input tensors on host (X: {type(X).__name__}, y: {type(y).__name__ if y is not None else 'None'}). "
                    f"Transfer your arrays using cp.asarray() before calling train/evaluate/predict."
                )
        else:  # numpy
            if is_x_cupy or is_y_cupy:
                raise TypeError(
                    f"[aether] Device mismatch: Model is configured for NumPy backend ('numpy'), "
                    f"but received CuPy GPU tensors. Transfer your data to host memory via cp.asnumpy() "
                    f"or migrate the model first via model.to('cupy')."
                )

    def forward(self, X, training=True):
        """
        Sequentially propagate input batch through all layers.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, loss_dinputs):
        """
        Sequentially propagate loss gradients backward through layers in reverse order.
        """
        dinputs = loss_dinputs
        for layer in reversed(self.layers):
            dinputs = layer.backward(dinputs)
        return dinputs

    def train(
        self,
        X,
        y,
        *,
        epochs=1,
        batch_size=None,
        shuffle=True,
        print_every=1,
        verbose = 1,
        validation_data=None,
        fit_preprocessor: bool | None = None
    ):
        """
        Train the compiled neural network on dataset (X, y).

        Resolves conditonal branches and step functions prior to the training loop,
        avoiding Python-level conditionals inside the batch loop, keeping CPU overhead low.

        If a preprocessing pipeline is attached, `X` may be passed as raw data.
        Without a pipeline, `X` and `y` must already reside on the model's device.

        Args:
            X (ndarray): Input features (NumPy or CuPy ndarray). Raw/un-preprocessed
                when a preprocessing pipeline is attached.
            y (ndarray): Target labels or one-hot ground truths.
            epochs (int): Number of full passes over the dataset.
            batch_size (int, optional): Mini-batch sample size. Defaults to None (full-batch).
            shuffle (bool, optional): Permutes sample indices each epoch without copying.
                Defaults to True.
            print_every (int): Step interval frequency for logging telemetry.
            verbose (int, optional): Verbosity mode for training telemetry.
                - `0`: Silent mode (no output printed).
                - `1`: Dynamic graphical progress bar with metrics (default).
                - `2`: Plain text summary per epoch without carriage returns/ANSI escapes.
            validation_data (tuple, optional): Raw (X_val, y_val) tuple evaluated per epoch.
            fit_preprocessor (bool, optional): Whether to fit the preprocessor
                (None=only if unfitted, True=always, False=never).
        Raises:
            RuntimeError: If the model has not been finalized, if no loss is configured,
                or if the attached preprocessing pipeline is still unfitted after this
                call (e.g. `fit_preprocessor=False` was passed on a never-fitted pipeline).
            TypeError: With no pipeline attached, if `X`/`y` (or `validation_data`) do
                not match the model's device backend. With a pipeline attached, if the
                pipeline emits arrays on a device other than the model's.
        Warns:
            UserWarning: If the preprocessor was already fitted and this call re-fits
                it anyway (e.g. `fit_preprocessor=True` on an already-fitted pipeline),
                discarding its previous statistics.
        Note:
            - Full-dataset preprocessor fitting temporarily materializes data in memory.
            - Do not pass manually pre-transformed data if a pipeline is attached to avoid
              duplicate transformations.
        """
        self._check_ready("training")
        if self.loss is None:
            raise RuntimeError(
                "[aether] Cannot train a model without a loss function."
                "Pass in a valid loss function to model.configure(loss=...) before finalize & train"
            )

        has_pipeline = self._has_pipeline()
        if not has_pipeline:
            self._assert_device_alignment(X, y)
            if validation_data is not None:
                self._assert_device_alignment(validation_data[0], validation_data[1])

        was_already_fitted = self.preprocessor.is_fitted
        should_fit = (
            not was_already_fitted if fit_preprocessor is None else fit_preprocessor
        )
        if should_fit:
            if was_already_fitted:
                warnings.warn(
                    "[aether] The attached preprocessing pipeline was already fitted, "
                    "but train() is re-fitting it on this call's data, discarding its "
                    "previous statistics. Pass fit_preprocessor=False to keep the "
                    "existing fit, or ignore this warning if refitting was intentional.",
                    UserWarning,
                    stacklevel=2,
                )
            self.preprocessor.fit(X)

        if not self.preprocessor.is_fitted:
            raise RuntimeError(
                "[aether] The attached preprocessing pipeline is not fitted, so training "
                "would run on untransformed data. Fit the pipeline before attaching it "
                "(e.g. Compose([...]).fit(X_train)), pass fit_preprocessor=True to fit it "
                "now, or omit fit_preprocessor to let train() fit it automatically."
            )

        if has_pipeline:
            self._assert_pipeline_device(X)

        xp = config.get_array_module(X)

        num_samples = len(X)
        effective_batch_size = batch_size if batch_size is not None else num_samples
        train_steps = (num_samples + effective_batch_size - 1) // effective_batch_size

        batch_slices = [
            (step * effective_batch_size, min((step + 1) * effective_batch_size, num_samples))
            for step in range(train_steps)
        ]

        if shuffle:
            def get_batch(epoch_indices, start_idx, end_idx):
                batch_idx = epoch_indices[start_idx:end_idx]
                return X[batch_idx], y[batch_idx]
        else:
            def get_batch(epoch_indices, start_idx, end_idx):
                return X[start_idx:end_idx], y[start_idx:end_idx]

        prepare_input = self.preprocessor.transform
        prepare_target = self._make_target_preparer(y)

        forward_fn = self.forward
        backward_fn = self.backward
        loss = self.loss
        accuracy = self.accuracy
        optimizer_obj = self.optimizer
        step_optimizer = self._step_optimizer
        advance_rng = self._rng_clock.advance
        has_reg = hasattr(loss, "regularization_loss") and any(
            getattr(layer, "weight_regularizer_l1", 0.0) > 0.0
            or getattr(layer, "weight_regularizer_l2", 0.0) > 0.0
            or getattr(layer, "bias_regularizer_l1", 0.0) > 0.0
            or getattr(layer, "bias_regularizer_l2", 0.0) > 0.0
            for layer in getattr(self, "trainable_layers", [])
        )
        reg_loss_fn = loss.regularization_loss if has_reg else None

        if has_reg:
            def run_step(batch_X, batch_y):
                advance_rng()
                out = forward_fn(batch_X, training=True)
                data_l = loss.calculate(out, batch_y)
                reg_l = reg_loss_fn()
                acc_l = accuracy.calculate(out, batch_y)
                loss.backward(out, batch_y)
                backward_fn(loss.dinputs)
                step_optimizer()
                return data_l, reg_l, acc_l
        else:
            def run_step(batch_X, batch_y):
                advance_rng()
                out = forward_fn(batch_X, training=True)
                data_l = loss.calculate(out, batch_y)
                acc_l = accuracy.calculate(out, batch_y)
                loss.backward(out, batch_y)
                backward_fn(loss.dinputs)
                step_optimizer()
                return data_l, 0.0, acc_l

        get_lr = lambda: getattr(optimizer_obj, "current_lr", getattr(optimizer_obj, "lr", None))

        progress = make_progress(verbose, train_steps, epochs, has_reg)

        for epoch in range(1, epochs + 1):
            loss.new_pass()
            accuracy.new_pass()
            progress.start_epoch(epoch)

            epoch_indices = xp.random.permutation(num_samples) if shuffle else None

            for step, (start_idx, end_idx) in enumerate(batch_slices):
                batch_X, batch_y = get_batch(epoch_indices, start_idx, end_idx)

                data_loss, reg_loss, acc_val = run_step(
                    prepare_input(batch_X), prepare_target(batch_y)
                )

                # Bar-only tick: pure host-side math, zero GPU sync.
                progress.tick(step + 1)

                # Telemetry: GPU-to-Host sync barrier strictly confined to log steps.
                if print_every and (step % print_every == 0 or step == train_steps - 1):
                    s_data_loss = float(data_loss)
                    s_reg_loss = float(reg_loss)
                    s_acc = float(acc_val)
                    lr = get_lr()

                    progress.update_metrics(
                        step + 1,
                        s_data_loss + s_reg_loss,
                        s_acc,
                        float(lr) if lr is not None else 0.0,
                        reg_loss=s_reg_loss if has_reg else None,
                    )

            accum = loss.calculate_accumulated(include_regularization=has_reg)
            epoch_loss = float(accum[0] + accum[1]) if isinstance(accum, tuple) else float(accum)
            epoch_acc = float(accuracy.calculate_accumulated())

            lr = get_lr()
            progress.commit_epoch(epoch, epoch_loss, epoch_acc, float(lr) if lr is not None else 0.0)

            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
                progress.commit_validation(val_loss, val_acc)

        progress.close()

    def evaluate(self, X, y, *, batch_size=None, verbose=1):
        """
        Evaluate the model's loss and metrics on validation/test data in inference mode.

        If a preprocessing pipeline is attached, `X` may be passed as raw data; its
        transforms are applied per mini-batch, and `y` is migrated automatically.
        Without a pipeline, `X` and `y` must already reside on the model's device.

        Args:
            X (ndarray): Evaluation input features (NumPy or CuPy ndarray).
                Raw/un-preprocessed when a preprocessing pipeline is attached.
            y (ndarray): Ground truth labels or targets.
            batch_size (int, optional): Mini-batch size. Defaults to None (full-batch).
            verbose (int | bool, optional): Verbosity mode for evaluation.
                - `0` / `False`: Silent mode (no output printed).
                - `1` / `True`: Prints the final evaluation loss and accuracy metrics.
                Defaults to 1.
        Returns:
            tuple: (val_loss, val_acc) representing the computed validation metrics.
        Raises:
            RuntimeError: If model has not been explicitly finalized before calling
                evaluate(), or if the attached preprocessing pipeline is unfitted.
            TypeError: If input/pipeline devices do not match backend configuration.
        Note:
            Do not pass manually pre-transformed data if a pipeline is attached to avoid
            duplicate transformations.
        """
        self._check_ready("evaluation", need_fitted=True)
        if self._has_pipeline():
            self._assert_pipeline_device(X)
        else:
            self._assert_device_alignment(X, y)
        num_samples = len(X)
        effective_batch_size = batch_size if batch_size is not None else num_samples
        eval_steps = (num_samples + effective_batch_size - 1) // effective_batch_size


        has_loss = self.loss is not None
        has_acc = self.accuracy is not None

        if has_loss:
            self.loss.new_pass()
        if has_acc:
            self.accuracy.new_pass()

        # Cache local forward function
        forward_fn = self.forward
        loss = self.loss
        accuracy = self.accuracy
        prepare_input = self.preprocessor.transform
        prepare_target = self._make_target_preparer(y)


        for step in range(eval_steps):
            start_idx = step * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, num_samples)

            batch_X = prepare_input(X[start_idx:end_idx])
            batch_y = prepare_target(y[start_idx:end_idx])

            # Forward propagation in evaluation mode
            output = forward_fn(batch_X, training=False)

            # Accumulate batch metrics without tracking gradients or regularization
            if has_loss:
                loss.calculate(output, batch_y, training=False)

            if has_acc:
                accuracy.calculate(output, batch_y)


        val_loss = float(loss.calculate_accumulated(include_regularization=False)) if has_loss else 0.0
        val_acc = float(accuracy.calculate_accumulated()) if has_acc else 0.0

        if verbose: 
            print(f"[Validation] loss: {val_loss:.4f} - acc: {val_acc:.4f}")

        return val_loss, val_acc

    def predict(self, X, *, batch_size=None, return_logits=False, stream_to_host=True):
        """
        Run batch inference on inputs and return model predictions.

        Automatically routes raw outputs through any fused loss activation
        (e.g., SoftMax in SoftmaxCategoricalCrossEntropy) unless raw logits are requested.

        When a preprocessing pipeline is attached, `X` may be raw and un-preprocessed.
        Without the pipeline, `X` must already match the model's target device.

        Args:
            X (ndarray): Input feature batch (NumPy or CuPy array).
                Raw/un-preprocessed when a preprocessing pipeline is attached.
            batch_size (int, optional): Mini-batch sample size. Defaults to None (full-batch).
            return_logits (bool, optional): If True, returns raw network logits
                even if an output activation was fused into the loss. Defaults to False.
            stream_to_host (bool, optional): If True, incrementally transfers batch
                outputs to host memory (NumPy) to minimize GPU VRAM consumption.
                Defaults to True.

        Returns:
            ndarray: Model predictions (probabilities or stacked batch outputs).

        Raises:
            RuntimeError: If model has not been explicitly finalized before calling
                predict(), or if the attached preprocessing pipeline is unfitted.
            TypeError: With no pipeline attached, if `X` does not match the model's
                device backend. With a pipeline attached, if the pipeline emits
                arrays on a device other than the model's.
        Note:
            Do not pass manually pre-transformed data if a pipeline is attached to avoid
            duplicate transformations.
        """
        self._check_ready("prediction", need_fitted=True)
        if self._has_pipeline():
            self._assert_pipeline_device(X)
        else:
            self._assert_device_alignment(X)

        num_samples = len(X)
        effective_batch_size = batch_size if batch_size is not None else num_samples
        prediction_steps = (num_samples + effective_batch_size - 1) // effective_batch_size

        forward_fn = self.forward
        prepare_input = self.preprocessor.transform
        activation = self._predict_activation
        apply_activation = (not return_logits) and (activation is not None)

        output_buffer = None

        for step in range(prediction_steps):
            start_idx = step * effective_batch_size
            end_idx = min(start_idx + effective_batch_size, num_samples)

            batch_output = forward_fn(prepare_input(X[start_idx:end_idx]), training=False)
            if apply_activation:
                act_out = activation.forward(batch_output, training=False)
                batch_output = act_out if act_out is not None else activation.output

            if output_buffer is None:
                output_module = np if stream_to_host else config.get_array_module(batch_output)
                output_buffer = output_module.empty(
                    (num_samples,) + batch_output.shape[1:], dtype=batch_output.dtype
                )

            output_buffer[start_idx:end_idx] = (
                config.to_device(batch_output, target="numpy") if stream_to_host else batch_output
            )

        return output_buffer

    def save(self, filepath: str):
        """
        Saves the model architecture and parameters to a single .aether archive.

        The manifest records the loss, optimizer, and accuracy metric under
        "compile", and any attached preprocessing pipeline under a top-level
        "preprocessor" key, so a loaded model can be fed raw, un-preprocessed
        data straight away. A model with no pipeline attached writes a null
        entry instead.

        Args:
            filepath (str): Path to output file (e.g. 'cifar10.aether').
        Raises:
            RuntimeError: If model is not finalized.
            ImportError: If safetensors is not installed.
            FileNotFoundError: If the directory being saved to does not exist.
        """
        if not self.is_finalized:
            raise RuntimeError(
                "[aether] Cannot save an unfinalized model. "
                "Call model.finalize(input_shape) first."
            )

        if safetensors is None:
            raise ImportError(
                "[aether] 'safetensors' is required for model serialization. "
                "Install it with `pip install safetensors`."
            )

        path = Path(filepath)
        if not path.parent.exists():
            raise FileNotFoundError(
                f"[aether] Directory '{path.parent}' does not exist. "
                f"Please create the directory before saving the model."
            )
        if path.suffix != ".aether":
            path = path.with_name(path.name + ".aether")

        # Capture Top-Level Config & Layer Manifest
        input_shape = self.layers[0].input_shape
        precision_str = (
            self.precision_policy.compute_dtype_name
            if getattr(self, "precision_policy", None)
            else None
        )

        arch_manifest = {
            "schema_version": _SCHEMA_VERSION,
            "input_shape": list(input_shape),
            "seed": self._seed,
            "precision_policy": precision_str,
            "compile":{
                "loss": _serialize_component(self.loss),
                "optimizer": _serialize_component(self.optimizer),
                "accuracy": _serialize_component(self.accuracy),
            },
            "preprocessor": _serialize_component(self.preprocessor),
            "layers": [
                {
                    "index": idx,
                    "class_name": type(layer).__name__,
                    "config": layer.get_config(),
                }
                for idx, layer in enumerate(self.layers)
            ],
        }

        weight_dict = {}
        for idx, layer in enumerate(self.layers):
            for name, tensor in layer.get_parameters().items():
                if tensor is None:
                    continue
                cpu_array = config.to_device(tensor, target="numpy")
                weight_dict[f"{idx}.{name}"] = np.ascontiguousarray(cpu_array)

        weights_bytes = safetensors.numpy.save(weight_dict)
        arch_json_bytes = json.dumps(arch_manifest, indent=2).encode("utf-8")

        with zipfile.ZipFile(path, "w") as zipf:
            zipf.writestr(
                "architecture.json", arch_json_bytes, compress_type=zipfile.ZIP_DEFLATED
            )
            zipf.writestr(
                "weights.safetensors", weights_bytes, compress_type=zipfile.ZIP_STORED
            )

    @classmethod
    def load(cls, filepath: str, device: str | None = None) -> "Model":
        """Loads a model architecture, training components, and parameters from an
        .aether zip archive.

        Args:
            filepath (str): Path to the saved .aether model file.
            device (str, optional): Target hardware device ('numpy' or 'cupy').
                Defaults to None (uses current active backend).

        Returns:
            Model: A configured, finalized, and initialized Model instance.

        Raises:
            ImportError: If safetensors is not installed.
            FileNotFoundError: If the archive does not exist.
            ValueError: If the manifest was written by a newer schema major version,
                or references an unknown layer/component class.

        Warns:
            UserWarning: If the manifest carries a newer schema minor version.
                Minor bumps are additive, so the archive still loads, but any key
                this build does not recognize is ignored.

        Example:
            >>> import aether as ae
            ... model = ae.Model.load("saved_models/cifar10_3block_cnn.aether")
            ... loss, acc = model.evaluate(X=X_test, y=y_test)
        """
        if safetensors is None:
            raise ImportError(
                "[aether] 'safetensors' is required to load models. "
                "Install it with `pip install safetensors`."
            )

        path = Path(filepath)
        if not path.exists():
            candidate = path.with_suffix(".aether") if not path.suffix else path.with_name(path.name + ".aether")
            if candidate.exists():
                path = candidate
            else:
                raise FileNotFoundError(f"[aether] Archive '{path}' does not exist.")

        active_backend = "cupy" if getattr(config.xp, "__name__", "") == "cupy" else "numpy"
        target_device = device if device is not None else active_backend

        with zipfile.ZipFile(path, "r") as zipf:
            arch_bytes = zipf.read("architecture.json")
            weights_bytes = zipf.read("weights.safetensors")

        manifest = json.loads(arch_bytes.decode("utf-8"))
        raw_weights = safetensors.numpy.load(weights_bytes)

        schema_version = str(manifest.get("schema_version", "1.0"))
        major_str, _, minor_str = schema_version.partition(".")
        try:
            schema_major = int(major_str)
        except ValueError:
            schema_major = _SCHEMA_MAJOR
        try:
            schema_minor = int(minor_str.split(".", 1)[0])
        except ValueError:
            schema_minor = 0

        if schema_major > _SCHEMA_MAJOR:
            raise ValueError(
                f"[aether] Archive '{path.name}' uses manifest schema v{schema_version}, "
                f"which requires a newer version of aether."
            )

        if schema_major == _SCHEMA_MAJOR and schema_minor > _SCHEMA_MINOR:
            warnings.warn(
                f"[aether] Archive '{path.name}' uses manifest schema v{schema_version}, "
                f"newer than the v{_SCHEMA_VERSION} this build of aether writes. It loads, "
                f"but may carry configuration this version does not recognize and will ignore.",
                UserWarning,
                stacklevel=2,
            )

        model = cls()

        if manifest.get("seed") is not None:
            model.manual_seed(manifest["seed"])

        # Reconstruct Layers (Inline, tuple-sanitized, subclass-checked)
        for layer_entry in manifest.get("layers", []):
            class_name = layer_entry["class_name"]
            cfg = layer_entry.get("config", {})

            sanitized_cfg = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in cfg.items()
            }

            layer_cls = getattr(layer_module, class_name, None)
            if layer_cls is None:
                raise ValueError(f"[aether] Unknown layer class '{class_name}' found in saved manifest.")
            if layer_cls is Layer or not (isinstance(layer_cls, type) and issubclass(layer_cls, Layer)):
                raise TypeError(f"[aether] Manifest entry '{class_name}' is not a valid concrete 'Layer' subclass.")

            model.add(layer_cls(**sanitized_cfg))

        # Reconstruct Training Components (Via table-driven helper)
        compile_cfg = manifest.get("compile") or {}
        components = {
            kind: _resolve_component(compile_cfg.get(kind), kind)
            for kind in ("loss", "optimizer", "accuracy")
        }

        components["preprocessor"] = transforms_module.deserialize(
            manifest.get("preprocessor")
        )

        if any(component is not None for component in components.values()):
            model.configure(**components)
        model.to(target_device)

        input_shape = tuple(manifest["input_shape"])
        model.finalize(input_shape=input_shape)

        # Load & Map Weights
        layer_param_map: dict[int, dict] = {}
        for flat_key, array in raw_weights.items():
            if "." not in flat_key:
                continue
            idx_str, param_name = flat_key.split(".", 1)
            idx = int(idx_str)
            if idx not in layer_param_map:
                layer_param_map[idx] = {}

            # Route arrays directly to the resolved target device
            layer_param_map[idx][param_name] = config.to_device(
                array, target=target_device
            )

        for idx, params in layer_param_map.items():
            if idx < len(model.layers):
                model.layers[idx].set_parameters(**params)

        if manifest.get("precision_policy") is not None:
            model.set_precision(manifest["precision_policy"])

        return model