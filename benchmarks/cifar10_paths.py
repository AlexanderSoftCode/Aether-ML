"""
Train the same CIFAR-10 CNN on three dispatch paths, for the README's throughput and
convergence figures.

Run from the repo root (see benchmarks/README.md for the full release workflow):

    python3 benchmarks/cifar10_paths.py benchmarks/history/v1.2.0/paths.json

--fused-only skips the slower paths, for adding another GPU to the convergence figure.

The paths, all built by cifar10_breakdown.build_model:

    numpy-cpu      the framework on the CPU fallback path
    cupy-fallback  tensors on the GPU, but every layer and the optimizer on that same
                   fallback path, so vectorized CuPy array ops and no custom kernels
    cupy-fused     the normal model.to("cupy") path

The precision policy is the same everywhere, but only the fused path's Conv2d can honor
fp16; the fallback paths warn and compute Conv2d in fp32.

Each run warms up, records loss and elapsed time for one epoch, then times BASELINE_STEPS
with no per-step sync. The epoch syncs after every step so each loss gets a timestamp,
which costs time on a fast step, so throughput comes from the unsynced block. The first
full-length train() runs slow (see cifar10_breakdown.main), and the epoch absorbs that.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import cupy as cp
from cifar10_breakdown import (
    BASELINE_STEPS,
    BATCH_SIZE,
    WARMUP_STEPS,
    bound_paths,
    build_model,
    load_train,
    run_meta,
    spread,
    timed_train,
)

LABELS = {
    "numpy-cpu": "NumPy CPU",
    "cupy-fallback": "CuPy, no custom kernels",
    "cupy-fused": "CuPy, fused kernels",
}
NUMPY_REPEATS = 1  # about 11 minutes a run, and the CPU is the least noisy path


def use_fallback_kernels(model):
    """Point a finalized GPU model's dispatch at the fallback path, leaving tensors on
    the GPU. The fallbacks take their array module from the data, so they run as CuPy."""
    for component in (*model.layers, model.loss, model.accuracy):
        if hasattr(component, "_compile_for_device"):
            component._compile_for_device("numpy")
    # Not Adam's _compile_for_device, which also moves its buffers to the host.
    model.optimizer._step_impl = model.optimizer._step_fallback
    # BatchNorm just lost its fused activation, so the skipped ReLUs have to come back.
    model._bind_fusion()


def prepare(key, X):
    model, _ = build_model(X, device="numpy" if key == "numpy-cpu" else "cupy")
    if key == "cupy-fallback":
        use_fallback_kernels(model)
    return model


def trace_epoch(model, X, y):
    """Per-step data loss and cumulative seconds over one epoch of the real train()."""
    calculate, step_optimizer = model.loss.calculate, model._step_optimizer
    pending_loss = None
    losses, elapsed_s = [], []
    total = 0.0

    def remember_loss(*args, **kwargs):
        nonlocal pending_loss
        pending_loss = calculate(*args, **kwargs)
        return pending_loss

    def end_step():
        nonlocal total, step_start
        step_optimizer()
        cp.cuda.Device().synchronize()
        total += time.perf_counter() - step_start
        elapsed_s.append(total)
        losses.append(float(pending_loss))
        step_start = time.perf_counter()  # the host readback above isn't step time

    model.loss.calculate = remember_loss
    model._step_optimizer = end_step
    try:
        step_start = time.perf_counter()
        model.train(X, y, epochs=1, batch_size=BATCH_SIZE, shuffle=True, verbose=0,
                    fit_preprocessor=False)
    finally:
        del model.loss.calculate  # it was a plain class method
        model._step_optimizer = step_optimizer
    return losses, elapsed_s


def run_path(key, X, y, repeats):
    steps_per_epoch = len(X) // BATCH_SIZE
    epoch_X, epoch_y = X[:steps_per_epoch * BATCH_SIZE], y[:steps_per_epoch * BATCH_SIZE]
    ms_per_step = []
    for _ in range(repeats):
        model = prepare(key, X)
        timed_train(model, X, y, WARMUP_STEPS)
        losses, elapsed_s = trace_epoch(model, epoch_X, epoch_y)
        ms_per_step.append(timed_train(model, X, y, BASELINE_STEPS) * 1e3 / BASELINE_STEPS)
        paths = bound_paths(model)
        del model
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    return {
        "key": key,
        "label": LABELS[key],
        "repeats": repeats,
        "ms_per_step": spread(ms_per_step),
        "loss": losses,  # from the last repeat
        "elapsed_s": elapsed_s,
        "paths": paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("output", type=Path, help="results file, for benchmarks/plot.py")
    parser.add_argument("--repeats", type=int, default=3,
                        help=f"runs per GPU path (default: 3); NumPy runs {NUMPY_REPEATS}")
    parser.add_argument("--fused-only", action="store_true",
                        help="run only cupy-fused, for a second GPU on the convergence figure")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    X, y = load_train()
    X_gpu, y_gpu = cp.asarray(X), cp.asarray(y)
    results = [run_path("cupy-fused", X_gpu, y_gpu, args.repeats)]
    if not args.fused_only:
        results = [
            run_path("numpy-cpu", X, y, NUMPY_REPEATS),
            run_path("cupy-fallback", X_gpu, y_gpu, args.repeats),
            *results,
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"meta": run_meta(), "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
