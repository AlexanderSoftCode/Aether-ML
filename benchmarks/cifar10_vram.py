"""
Record transient VRAM over two training steps, fused kernels against the fallback path,
for the README's VRAM figure.

Run from the repo root (see benchmarks/README.md for the full release workflow):

    python3 benchmarks/cifar10_vram.py benchmarks/history/v1.2.0/vram.json

A CuPy memory hook logs live pool bytes at every malloc and free inside the real
model.train(). "Transient" is the delta above the bytes already live after warm-up
(parameters, AdamW state, the uploaded batches). The train() call records one extra step
that is thrown away, which leaves its one-off setup out of the trace and gives the first
kept step a start mark. The batch gather and preprocessing land in the forward phase.

Allocation order doesn't depend on timing, so repeats must produce identical traces;
anything else means the path is nondeterministic and the run fails.
"""
import argparse
import gc
import itertools
import json
from pathlib import Path

import cupy as cp
import cupy.cuda.memory_hook
from cifar10_breakdown import BATCH_SIZE, WARMUP_STEPS, load_train, run_meta
from cifar10_paths import LABELS, prepare

RECORDED_STEPS = 2
REPEATS = 3
# Share of a step's x axis per phase. Layout only: these are not durations.
PHASE_WIDTHS = {"forward": 0.4, "loss": 0.1, "backward": 0.4, "optimizer": 0.1}


class LivePoolTrace(cupy.cuda.memory_hook.MemoryHook):
    """Live bytes since the hook was entered, appended at every malloc and free."""

    name = "AetherLivePoolTrace"

    def __init__(self):
        super().__init__()
        self.live = 0
        self.trace = []

    def malloc_postprocess(self, **kwargs):
        self.live += kwargs["mem_size"]
        self.trace.append(self.live)

    def free_postprocess(self, **kwargs):
        self.live -= kwargs["mem_size"]
        self.trace.append(self.live)


def record_steps(model, X, y):
    """(trace, seams): the live-bytes trace of train(), and the trace index at the end
    of every phase. The phases end at the last call of each, in PHASE_WIDTHS order."""
    recorder = LivePoolTrace()
    seams = []

    def marked(func):
        def wrapper(*args, **kwargs):
            out = func(*args, **kwargs)
            seams.append(len(recorder.trace))
            return out
        return wrapper

    instance_attrs = ((model, "forward"), (model.loss, "backward"), (model, "backward"),
                      (model, "_step_optimizer"))
    originals = [(obj, attr, vars(obj).get(attr)) for obj, attr in instance_attrs]
    for obj, attr in instance_attrs:
        setattr(obj, attr, marked(getattr(obj, attr)))
    try:
        n = (RECORDED_STEPS + 1) * BATCH_SIZE
        with recorder:
            model.train(X[:n], y[:n], epochs=1, batch_size=BATCH_SIZE, shuffle=False,
                        verbose=0, fit_preprocessor=False)
    finally:
        for obj, attr, original in originals:
            if original is None:
                delattr(obj, attr)  # it was a plain class method
            else:
                setattr(obj, attr, original)
    return recorder.trace, seams


def to_phase_axis(trace, seams):
    """Keep the recorded steps and spread each phase's events evenly over its width, so
    both paths share an x axis. Returns (x, transient bytes, phase) per event."""
    phases = len(PHASE_WIDTHS)
    kept = seams[phases - 1:]  # from the end of the discarded step
    x, transient, phase_of = [], [], []
    bands = itertools.cycle(PHASE_WIDTHS.items())
    position = 0.0
    for (lo, hi), (phase, width) in zip(itertools.pairwise(kept), bands):
        for i in range(lo, hi):
            x.append(position + width * (i - lo) / (hi - lo))
            transient.append(trace[i])
            phase_of.append(phase)
        position += width
    return x, transient, phase_of


def measure(key, X, y):
    model = prepare(key, X)
    model.train(X, y, epochs=1, batch_size=BATCH_SIZE, shuffle=False, verbose=0,
                fit_preprocessor=False)
    cp.cuda.Device().synchronize()
    baseline = cp.get_default_memory_pool().used_bytes()

    trace, seams = record_steps(model, X, y)
    x, transient, phase_of = to_phase_axis(trace, seams)
    peak = max(range(len(transient)), key=transient.__getitem__)

    del model
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "key": key,
        "label": LABELS[key],
        "baseline_bytes": baseline,
        "peak_transient_bytes": transient[peak],
        "peak_x": x[peak],
        "peak_phase": phase_of[peak],
        "x": x,
        "transient_bytes": transient,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("output", type=Path, help="results file, for benchmarks/plot.py")
    args = parser.parse_args()

    # Only what the run consumes goes on the GPU, or the baseline is mostly dataset.
    n = WARMUP_STEPS * BATCH_SIZE
    X, y = (cp.asarray(array[:n]) for array in load_train())

    results = []
    for key in ("cupy-fused", "cupy-fallback"):
        runs = [measure(key, X, y) for _ in range(REPEATS)]
        if any(run["transient_bytes"] != runs[0]["transient_bytes"] for run in runs):
            raise RuntimeError(f"[aether] {key}: VRAM traces differ across {REPEATS} repeats.")
        results.append(runs[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "meta": {**run_meta(), "recorded_steps": RECORDED_STEPS, "repeats": REPEATS},
        "phase_widths": PHASE_WIDTHS,
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
