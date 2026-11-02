"""
Train the v1 CIFAR-10 CNN in PyTorch, as the reference bar on the release figure.

Run from the repo root, in its own process: PyTorch and CuPy can't share one (see
benchmarks/README.md). PyTorch isn't a dependency of the package; install the build for your
GPU from pytorch.org.

    python3 benchmarks/cifar10_torch.py benchmarks/history/v1.2.0/torch_rx9070xt.json

The model is build_model in cifar10_breakdown.py, written the way a PyTorch user would, with
the whole step under torch.compile: default NCHW layout, fp16 autocast and no loss scaling.
Autocast also keeps the activations after each Conv2d in fp16, where Aether keeps them fp32.
Every step does what Aether's does: standardize the batch on the GPU, forward, label-smoothed
loss plus the L2 term on the third Conv2d, accuracy, backward and an AdamW step. Aether's
learning-rate decay is left out, because it changes the loss and not the time per step.

The timing protocol is cifar10_breakdown.py's baseline: warm up, discard one full-length run,
then time BASELINE_STEPS per repeat with no per-step sync. Compilation happens in the warm-up,
so it counts toward warmup_s and not baseline_ms.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from common import (
    BASELINE_STEPS,
    BATCH_SIZE,
    WARMUP_STEPS,
    base_meta,
    load_train,
    spread,
)

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    sys.exit("[aether] PyTorch is not installed; install the build for your GPU from pytorch.org.")

SEED = 42  # build_model's manual_seed
LABEL_SMOOTHING = 0.05
L2 = 1e-5  # on the third Conv2d only
PARAMETERS = 141_354  # 3 Conv2d, 5 BatchNorm (gamma, beta) and the Dense, counted by hand


def build_model():
    def conv_block(in_channels, out_channels):
        return [nn.Conv2d(in_channels, out_channels, 3, padding="same"),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),  # Aether's momentum is 0.9
                nn.ReLU()]

    model = nn.Sequential(
        *conv_block(3, 32), *conv_block(32, 32), nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
        *conv_block(32, 64), *conv_block(64, 64), nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),
        *conv_block(64, 128), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
    ).cuda()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != PARAMETERS:
        sys.exit(f"[aether] The PyTorch model has {parameters} parameters, not {PARAMETERS}; "
                 "it no longer matches build_model in cifar10_breakdown.py.")
    return model


def make_step(X):
    """A compiled training step on the GPU."""
    torch.manual_seed(SEED)
    model = build_model()
    l2_weight = model[8].weight  # the third Conv2d
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaled = X.float() / 255.0
    mean, std = scaled.mean(), scaled.std(correction=0)
    del scaled
    correct = torch.zeros((), device=X.device, dtype=torch.int64)

    def step(x, y):
        x = ((x.float() * (1.0 / 255.0) - mean) / std).permute(0, 3, 1, 2).contiguous()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x)
        logits = logits.float()
        loss = (F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
                + L2 * l2_weight.square().sum())
        correct.add_((logits.argmax(1) == y).sum())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss

    return torch.compile(step)


def timed_train(step, X, y, num_steps):
    """Wall time of num_steps shuffled full batches."""
    n = num_steps * BATCH_SIZE
    torch.cuda.synchronize()
    start = time.perf_counter()
    order = torch.randperm(n, device=X.device)
    for i in range(num_steps):
        batch = order[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        step(X[batch], y[batch])
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("output", type=Path, help="results file, for benchmarks/plot.py")
    parser.add_argument("--repeats", type=int, default=3,
                        help="baseline runs after warm-up (default: 3)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not torch.cuda.is_available():
        sys.exit("[aether] PyTorch can't see a GPU.")

    X, y = (torch.as_tensor(array).cuda() for array in load_train())
    step = make_step(X)

    warmup_s = timed_train(step, X, y, WARMUP_STEPS)
    timed_train(step, X, y, BASELINE_STEPS)  # the first full-length run is slow
    baseline_runs = [timed_train(step, X, y, BASELINE_STEPS) * 1e3 / BASELINE_STEPS
                     for _ in range(args.repeats)]

    result = {
        "meta": {
            **base_meta(Path(__file__).resolve().parent),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "mode": "compiled",
            "repeats": args.repeats,
            "warmup_s": warmup_s,
        },
        "baseline_ms": spread(baseline_runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
