"""
Per-phase time breakdown of one training step for the v1 CIFAR-10 CNN.

Run from the repo root, then diff two runs with benchmarks/compare.py:

    python3 benchmarks/cifar10_breakdown.py benchmarks/results/before.json
    python3 benchmarks/cifar10_breakdown.py benchmarks/results/after.json
    python3 benchmarks/compare.py benchmarks/results/before.json benchmarks/results/after.json

Prints nothing on success; everything goes to the JSON file. --repeats N (default 3)
sets how many baseline and breakdown runs follow the warm-up.

1. Baseline: model.train() untouched, wall-clock ms/step with no instrumentation.
2. Breakdown: the same model.train(), with every per-step call it makes (each
   transform, layer forward/backward, loss, accuracy, optimizer step) wrapped in a
   CUDA/HIP event pair and a host timestamp. Events mark the stream without
   syncing, so GPU time is attributed per phase; host time is how long Python took
   to dispatch that phase. Time outside the wrapped calls is the "unattributed" row.

Each phase stores the median, min and max over the repeats, so compare.py can tell
a real change from noise. The bound forward path of every layer is stored too, so a
silent fallback (e.g. conv leaving the WMMA path) shows up in the diff.

Since train() itself runs, changes to it (a skipped backward, fused nodes) show up
without editing this file. Calls it no longer makes drop out as removed rows; work
it does outside the wrapped methods grows "unattributed".
"""
import argparse
import datetime
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cupy as cp

import aether as ae

TRAIN_DIR = Path("data") / "cifar-10" / "cifar-10_train.npz"

BATCH_SIZE = 256
COMPUTE_DTYPE = "float16"
WARMUP_STEPS = 20
BASELINE_STEPS = 128
BREAKDOWN_STEPS = 100


def build_model(X_train):
    """Same model, pipeline and precision as examples/cifar10/v1_cifar10.py."""
    feature_pipeline = ae.Compose([
        ae.ToTensor(dtype='float32', target_device="cupy"),
        ae.Rescale(factor=1.0 / 255.0),
        ae.StandardScaler()
    ])

    model = ae.Model()
    model.manual_seed(seed=42)

    model.add(ae.Conv2d(3, 32, (3, 3), (1, 1), padding="same"))
    model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
    model.add(ae.ReLU())
    model.add(ae.Conv2d(32, 32, (3, 3), (1, 1), padding="same"))
    model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
    model.add(ae.ReLU())
    model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
    model.add(ae.SpatialDropout(rate=0.1, seed=42))

    model.add(ae.Conv2d(32, 64, (3, 3), (1, 1), padding="same", l2=1e-5))
    model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
    model.add(ae.ReLU())
    model.add(ae.Conv2d(64, 64, (3, 3), (1, 1), padding="same"))
    model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
    model.add(ae.ReLU())
    model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
    model.add(ae.SpatialDropout(rate=0.15, seed=42))

    model.add(ae.Conv2d(64, 128, (3, 3), (1, 1), padding="same"))
    model.add(ae.BatchNorm(epsilon=1e-5, momentum=0.9))
    model.add(ae.ReLU())
    model.add(ae.MaxPool2d((2, 2), (2, 2), padding="valid"))
    model.add(ae.SpatialDropout(rate=0.2, seed=42))

    model.add(ae.GlobalAvgPool())
    model.add(ae.Dense(128, 10))

    model.configure(
        loss=ae.SoftmaxCategoricalCrossEntropy(label_smoothing=0.05),
        optimizer=ae.AdamW(lr=0.001, decay=1e-4, weight_decay=0.01),
        accuracy=ae.CategoricalAccuracy(),
        preprocessor=feature_pipeline
    )

    model.to('cupy')
    model.set_precision(compute_dtype=COMPUTE_DTYPE)
    model.finalize(input_shape=X_train.shape[1:])
    feature_pipeline.fit(X_train)
    return model, feature_pipeline


def timed_train(model, X_train, y_train, num_steps):
    """Wall time of an untouched model.train() over num_steps full batches."""
    n = num_steps * BATCH_SIZE
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    model.train(X_train[:n], y_train[:n], epochs=1, batch_size=BATCH_SIZE,
                shuffle=True, verbose=0, fit_preprocessor=False)
    cp.cuda.Device().synchronize()
    return time.perf_counter() - start


def phases(model, transforms):
    """(row name, object, method name) for every call model.train() makes per step."""
    layers = model.layers
    loss = model.loss
    rows = [(f"preprocess  {type(t).__name__}", t, "transform") for t in transforms]
    rows += [(f"forward  {i:02d} {type(layer).__name__}", layer, "forward")
             for i, layer in enumerate(layers)]
    rows += [("loss.calculate", loss, "calculate"),
             ("loss.regularization_loss", loss, "regularization_loss"),
             ("accuracy.calculate", model.accuracy, "calculate"),
             ("loss.backward", loss, "backward")]
    rows += [(f"backward {i:02d} {type(layers[i]).__name__}", layers[i], "backward")
             for i in reversed(range(len(layers)))]
    # Last call of every step, so its end marks the step boundary below.
    rows += [("optimizer.step", model, "_step_optimizer")]
    return rows


def breakdown(model, transforms, X_train, y_train):
    """GPU and host ms per phase of the real model.train(), as {row name: (gpu, host)}.

    Every phase method is wrapped on its object for the duration of the call, so
    train() itself runs unmodified. The optimizer wrapper syncs once per step, which
    train() doesn't: without it Python runs ahead until the launch queue fills, and
    host time would measure waiting on the GPU instead of dispatch. The sync wait is
    left out of every row. Step 0 is skipped because it has no previous step end.
    Time between wrapped calls (batch gather, target preparation, anything train()
    does that isn't wrapped) goes to the "unattributed" row.
    """
    rows = phases(model, transforms)
    last = len(rows) - 1
    events = [[(cp.cuda.Event(), cp.cuda.Event()) for _ in rows]
              for _ in range(BREAKDOWN_STEPS)]
    host_ms = [[None] * len(rows) for _ in range(BREAKDOWN_STEPS)]
    step_end_host = [0.0] * BREAKDOWN_STEPS  # before the sync
    step_resume_host = [0.0] * BREAKDOWN_STEPS  # after it
    step = 0

    # train() also calls some phases outside its step loop: the pipeline once on X[:1]
    # to check its device, and regularization_loss for the epoch total. The first lands
    # in step 0, which is discarded anyway; the second runs after the last step.
    def wrap(slot, func):
        def timed(*args, **kwargs):
            nonlocal step
            if step == BREAKDOWN_STEPS:
                return func(*args, **kwargs)
            if step > 0 and host_ms[step][slot] is not None:
                raise RuntimeError(f"[aether] {rows[slot][0]} ran twice in one step; "
                                   "the breakdown assumes one call per step.")
            start, end = events[step][slot]
            start.record()
            t0 = time.perf_counter()
            out = func(*args, **kwargs)
            end.record()
            t1 = time.perf_counter()
            host_ms[step][slot] = (t1 - t0) * 1e3
            if slot == last:
                step_end_host[step] = t1
                end.synchronize()
                step_resume_host[step] = time.perf_counter()
                step += 1
            return out
        return timed

    originals = [(obj, attr, vars(obj).get(attr)) for _, obj, attr in rows]
    for slot, (_, obj, attr) in enumerate(rows):
        setattr(obj, attr, wrap(slot, getattr(obj, attr)))
    try:
        timed_train(model, X_train, y_train, BREAKDOWN_STEPS)
    finally:
        for obj, attr, original in originals:
            if original is None:
                delattr(obj, attr)  # it was a plain class method
            else:
                setattr(obj, attr, original)

    gpu = [0.0] * len(rows)
    host = [0.0] * len(rows)
    called = [False] * len(rows)
    gpu_rest = host_rest = 0.0
    for k in range(1, BREAKDOWN_STEPS):
        step_gpu = cp.cuda.get_elapsed_time(events[k - 1][last][1], events[k][last][1])
        step_host = (step_end_host[k] - step_resume_host[k - 1]) * 1e3
        for slot in range(len(rows)):
            if host_ms[k][slot] is None:
                continue
            g = cp.cuda.get_elapsed_time(*events[k][slot])
            gpu[slot] += g
            host[slot] += host_ms[k][slot]
            called[slot] = True
            step_gpu -= g
            step_host -= host_ms[k][slot]
        gpu_rest += step_gpu
        host_rest += step_host

    n = BREAKDOWN_STEPS - 1
    result = {name: (gpu[i] / n, host[i] / n)
              for i, (name, _, _) in enumerate(rows) if called[i]}
    result["unattributed"] = (gpu_rest / n, host_rest / n)
    return result


def spread(samples):
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples)}


def git_state():
    """(commit, dirty) of the checkout, or (None, None) outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def bound_paths(model):
    """Forward function each layer is bound to, keyed like '03 Conv2d'."""
    paths = {}
    for i, layer in enumerate(model.layers):
        func = getattr(layer.forward, "func", layer.forward).__name__  # pooling binds a partial
        if isinstance(layer, ae.Conv2d):
            func += f" compute={layer._effective_compute_dtype()}"
        paths[f"{i:02d} {type(layer).__name__}"] = func
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("output", type=Path, help="results file, for benchmarks/compare.py")
    parser.add_argument("--repeats", type=int, default=3,
                        help="baseline and breakdown runs after warm-up (default: 3)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if not TRAIN_DIR.exists():
        sys.exit(
            f"[aether] {TRAIN_DIR} not found. Run from the repo root after creating it "
            "with examples/cifar10/cifar-setup.ipynb."
        )
    with cp.load(TRAIN_DIR, allow_pickle=False) as data:
        X_train = data["X_train"]
        y_train = data["y_train"]

    model, feature_pipeline = build_model(X_train)
    transforms = feature_pipeline.transforms

    # Warm-up covers per-shape kernel JIT and the memory pool. The first full-length
    # train() still runs ~0.8 ms/step slow, so one baseline run is thrown away.
    warmup_s = timed_train(model, X_train, y_train, WARMUP_STEPS)
    timed_train(model, X_train, y_train, BASELINE_STEPS)
    baseline_runs, breakdown_runs = [], []
    for _ in range(args.repeats):
        baseline_runs.append(
            timed_train(model, X_train, y_train, BASELINE_STEPS) * 1e3 / BASELINE_STEPS)
        breakdown_runs.append(breakdown(model, transforms, X_train, y_train))

    commit, dirty = git_state()
    result = {
        "meta": {
            "commit": commit,
            "dirty": dirty,
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy": cp.__version__,
            "python": platform.python_version(),
            "batch_size": BATCH_SIZE,
            "compute_dtype": COMPUTE_DTYPE,
            "warmup_steps": WARMUP_STEPS,
            "baseline_steps": BASELINE_STEPS,
            "breakdown_steps": BREAKDOWN_STEPS,
            "repeats": args.repeats,
            "warmup_s": warmup_s,
        },
        "paths": bound_paths(model),
        "baseline_ms": spread(baseline_runs),
        "gpu_total_ms": spread([sum(g for g, _ in run.values()) for run in breakdown_runs]),
        "host_total_ms": spread([sum(h for _, h in run.values()) for run in breakdown_runs]),
        "rows": [
            {"name": name, "stage": name.split()[0],
             "gpu_ms": spread([run[name][0] for run in breakdown_runs]),
             "host_ms": spread([run[name][1] for run in breakdown_runs])}
            for name in breakdown_runs[0]
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
