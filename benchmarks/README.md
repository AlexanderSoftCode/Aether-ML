# Benchmarks

The figures in the main README and the release notes come from the scripts here, on one
CNN: `build_model` in `cifar10_breakdown.py`, the same model as
`examples/cifar10/v1_cifar10.py`.

| Script | What it measures | Figures |
|---|---|---|
| `cifar10_breakdown.py` | ms/step, split by training-step phase, on the normal GPU path | `benchmark_release` |
| `cifar10_paths.py` | ms/step and loss over one epoch, on NumPy, fallback CuPy and fused kernels | `benchmark_throughput`, `benchmark_convergence` |
| `cifar10_vram.py` | live VRAM over two steps, on fallback CuPy and fused kernels | `benchmark_vram` |
| `compare.py` | prints what changed between two `cifar10_breakdown.py` results | none |
| `plot.py` | draws the figures from saved results | all of the above |

The scripts need a GPU and CuPy. They also need CIFAR-10 in `data/cifar-10`: create it with
`examples/cifar10/cifar-setup.ipynb`. `plot.py` needs matplotlib instead, which isn't a
dependency of the package.

## Before and after a change

    python3 benchmarks/cifar10_breakdown.py benchmarks/results/before.json
    python3 benchmarks/cifar10_breakdown.py benchmarks/results/after.json
    python3 benchmarks/compare.py benchmarks/results/before.json benchmarks/results/after.json

`benchmarks/results/` is ignored by git, so it's the place for working numbers.

## Regenerating the figures for a release

Published numbers live in `benchmarks/history/<version>/`, so a figure can be redrawn
without rerunning anything. Measure from clean worktrees: the scripts stay in your
checkout, `PYTHONPATH` picks the `aether` being measured, and each result records that
tree's commit. `--relative-paths` lets the worktree resolve inside the devcontainer.

    git worktree add --relative-paths --detach .worktrees/old v1.1.0
    git worktree add --relative-paths --detach .worktrees/new <release commit>

Run from the repo root on each GPU, naming the file after that GPU:

    PYTHONPATH=.worktrees/old python3 benchmarks/cifar10_breakdown.py benchmarks/history/v1.1.0/breakdown_rtx3050.json
    PYTHONPATH=.worktrees/new python3 benchmarks/cifar10_breakdown.py benchmarks/history/v1.2.0/breakdown_rtx3050.json

On a GPU other than the one the README names, also add its fused path to the convergence
figure:

    PYTHONPATH=.worktrees/new python3 benchmarks/cifar10_paths.py --fused-only benchmarks/history/v1.2.0/paths_fused_rtx3050.json

Then once, on the machine the README names, for the path figures:

    PYTHONPATH=.worktrees/new python3 benchmarks/cifar10_vram.py benchmarks/history/v1.2.0/vram.json
    PYTHONPATH=.worktrees/new python3 benchmarks/cifar10_paths.py benchmarks/history/v1.2.0/paths.json

Draw them, then remove the worktrees:

    python3 benchmarks/plot.py v1.1.0 v1.2.0
    git worktree remove .worktrees/old
    git worktree remove .worktrees/new

`cifar10_paths.py` runs NumPy once and each GPU path three times, which takes about 15
minutes. The cross-version figure uses only `cifar10_breakdown.py`, so older versions need nothing
else from this folder. Each GPU gets its own panel and scale on it: comparing versions is
only a fair A/B on one machine.
