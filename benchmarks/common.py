"""
What the benchmark scripts share. It imports neither CuPy nor PyTorch, because the two
can't be loaded into one process (see benchmarks/README.md).
"""
import datetime
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

TRAIN_DIR = Path("data") / "cifar-10" / "cifar-10_train.npz"

BATCH_SIZE = 256
COMPUTE_DTYPE = "float16"
WARMUP_STEPS = 20
BASELINE_STEPS = 128


def load_train():
    """CIFAR-10 training images and labels, as host arrays."""
    if not TRAIN_DIR.exists():
        sys.exit(
            f"[aether] {TRAIN_DIR} not found. Run from the repo root after creating it "
            "with examples/cifar10/cifar-setup.ipynb."
        )
    with np.load(TRAIN_DIR, allow_pickle=False) as data:
        return data["X_train"], data["y_train"]


def spread(samples):
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples)}


def git_state(source):
    """(commit, dirty) of the checkout containing `source`, or (None, None) outside git,
    so a run against a worktree of an older tag records that tag."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, cwd=source).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, check=True, cwd=source).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def base_meta(source):
    """What a result file says about the run, whatever library ran it. `source` is the
    directory of the code that was measured."""
    commit, dirty = git_state(source)
    return {
        "commit": commit,
        "dirty": dirty,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "batch_size": BATCH_SIZE,
        "compute_dtype": COMPUTE_DTYPE,
        "warmup_steps": WARMUP_STEPS,
        "baseline_steps": BASELINE_STEPS,
    }
