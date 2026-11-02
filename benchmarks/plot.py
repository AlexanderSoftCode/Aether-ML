"""
Draw the README's benchmark figures from the results in benchmarks/history.

Run from the repo root, with matplotlib installed (the package doesn't depend on it):

    python3 benchmarks/plot.py v1.1.0 v1.2.0

Writes light and dark PNGs to notebooks/assets/benchmarks:

    benchmark_throughput, benchmark_convergence, benchmark_vram   from <new>/
    benchmark_release, old against new per GPU                    from both versions

Each GPU's release panel has its own scale and compares only with itself, so every panel
is a true A/B on one machine. A torch_<gpu>.json in <new>/ adds PyTorch's bar to that
GPU's panel; its bar is the wall-clock ms/step, the one number it has. On the convergence figure, any paths_fused_<gpu>.json adds
that GPU's fused path as a dashed curve; the title and equal-loss line stay on paths.json.
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

HISTORY = Path(__file__).resolve().parent / "history"
ASSETS = Path(__file__).resolve().parents[1] / "notebooks" / "assets" / "benchmarks"

plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DPI = 200
FIGSIZE = (8.0, 4.5)
SZ_TITLE, SZ_CAPTION, SZ_VALUE, SZ_UNIT = 14.0, 9.5, 21.0, 10.5
SZ_AXIS, SZ_TICK, SZ_FOOTER = 9.5, 10.0, 8.5
MIB = 1024.0 * 1024.0
TIMES = "×"
SEED = 42  # build_model's manual_seed

THEMES = {
    "light": {
        "suffix": "",
        "surface": "#fcfcfb",
        "legend_bg": "#f4f4f0",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "raw_alpha": 0.18,
        "series": {"cupy-fused": "#2a9d5c", "cupy-fallback": "#2a78d6", "numpy-cpu": "#4a3aa7",
                   "other-gpu-fused": "#5da938"},
        "stages": {"forward": "#2a9d5c", "backward": "#16643a"},
    },
    "dark": {
        "suffix": "-dark",
        "surface": "#1a1a19",
        "legend_bg": "#222220",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "raw_alpha": 0.38,
        "series": {"cupy-fused": "#3dbb74", "cupy-fallback": "#3987e5", "numpy-cpu": "#9085e9",
                   "other-gpu-fused": "#81d262"},
        "stages": {"forward": "#3dbb74", "backward": "#9fe8bd"},
    },
}
STAGE_LABELS = {"forward": "Forward", "backward": "Backward", "rest": "Everything else"}
THROUGHPUT_ORDER = ("numpy-cpu", "cupy-fallback", "cupy-fused")
VRAM_DRAW_ORDER = ("cupy-fallback", "cupy-fused")  # the taller area goes down first
MIN_LOSS_DROP = 0.25  # a curve that falls less than this didn't converge
DASHES = (0, (2, 1.2))  # in line widths; the legend handle ends on a dash, not a sliver
PHASE_LABEL_ROW ={"forward": 0, "loss": 1, "backward": 0, "optimizer": 1}


def fmt_ms(ms):
    return f"{ms:.0f}" if ms >= 100 else f"{ms:.1f}" if ms >= 10 else f"{ms:.2f}"


def fmt_time(seconds):
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s" if seconds >= 1 else f"{seconds:.2g} s"


def fmt_multiple(ratio):
    return f"{ratio:.0f}{TIMES}" if ratio >= 10 else f"{ratio:.1f}{TIMES}"


def fmt_mib(num_bytes):
    mib = num_bytes / MIB
    return f"{mib:,.0f}" if mib >= 100 else f"{mib:,.1f}"


def new_figure(theme, title, caption, footer, height=FIGSIZE[1]):
    """A figure with the title block and footer at the same size on any height."""
    fig = plt.figure(figsize=(FIGSIZE[0], height), dpi=DPI)
    fig.patch.set_facecolor(theme["surface"])
    fig.text(0.042, 1 - 0.2025 / height, title, fontsize=SZ_TITLE, color=theme["primary"],
             fontweight="bold", va="top")
    fig.text(0.042, 1 - 0.5535 / height, caption, fontsize=SZ_CAPTION, color=theme["secondary"], va="top")
    fig.text(0.5, 0.054 / height, footer, fontsize=SZ_FOOTER, color=theme["muted"], va="bottom",
             ha="center")
    return fig


def new_axes(fig, rect, theme):
    ax = fig.add_axes(rect)
    ax.set_facecolor(theme["surface"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(theme["axis"])
    ax.tick_params(colors=theme["secondary"], labelsize=SZ_TICK, length=0)
    ax.set_axisbelow(True)
    return ax


def footer_text(meta, extra=""):
    return (f"{meta['gpu']} · CuPy {meta['cupy']} · CIFAR-10 · "
            f"batch {meta['batch_size']} · seed {SEED} · {meta['commit'][:7]}{extra}")


# --- Throughput ---------------------------------------------------------------------

def figure_throughput(paths, theme):
    results = {r["key"]: r for r in paths["results"]}
    entries = [results[key] for key in THROUGHPUT_ORDER]
    fused_ms = results["cupy-fused"]["ms_per_step"]["median"]

    def delta(entry):
        if entry["key"] == "cupy-fused":
            return "baseline"
        return f"{fmt_multiple(entry['ms_per_step']['median'] / fused_ms)} slower"

    gain = results["cupy-fallback"]["ms_per_step"]["median"] / fused_ms
    fig = new_figure(
        theme,
        f"Fused kernels deliver {fmt_multiple(gain)} higher throughput on GPU",
        "Same CNN, same seed, same fp16 policy. Conv2d can only honor fp16 on the fused path.",
        footer_text(paths["meta"], f" · median of {results['cupy-fused']['repeats']} GPU runs"),
    )

    tiles = fig.add_axes([0.042, 0.585, 0.935, 0.215])
    tiles.set_xlim(0, len(entries))
    tiles.set_ylim(0, 1)
    tiles.axis("off")
    fig.canvas.draw()
    for i, entry in enumerate(entries):
        color = theme["series"][entry["key"]]
        tiles.plot([i + 0.035], [0.845], marker="o", markersize=8.5, color=color, clip_on=False)
        tiles.text(i + 0.105, 0.845, entry["label"], fontsize=SZ_CAPTION,
                   color=theme["secondary"], va="center")
        value = tiles.text(i, 0.38, fmt_ms(entry["ms_per_step"]["median"]), fontsize=SZ_VALUE,
                           color=theme["primary"], fontweight="bold")
        right = value.get_window_extent(fig.canvas.get_renderer()).x1
        tiles.text(tiles.transData.inverted().transform((right, 0))[0] + 0.024, 0.38, "ms",
                   fontsize=SZ_UNIT, color=theme["secondary"])
        tiles.text(i, 0.09, delta(entry), fontsize=SZ_CAPTION, color=theme["secondary"])
    fig.add_artist(plt.Line2D([0.042, 0.977], [0.556, 0.556], color=theme["grid"], linewidth=1.0))

    ax = new_axes(fig, [0.248, 0.190, 0.729, 0.318], theme)
    ax.set_xscale("log")
    ax.grid(axis="x", color=theme["grid"], linewidth=1.0)
    rows = np.arange(len(entries))[::-1]
    for row, entry in zip(rows, entries):
        ms = entry["ms_per_step"]["median"]
        color = theme["series"][entry["key"]]
        ax.plot([fused_ms, ms], [row, row], color=color, linewidth=2.0, solid_capstyle="round")
        ax.plot([ms], [row], marker="o", markersize=11, color=color,
                markeredgecolor=theme["surface"], markeredgewidth=2.0)
        ax.annotate(delta(entry), xy=(ms, row), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=SZ_TICK, color=theme["primary"])
    ax.set_yticks(rows, [entry["label"] for entry in entries], color=theme["primary"])
    ax.set_ylim(-0.72, len(entries) - 0.28)
    ax.set_xlim(fused_ms / 1.7, max(e["ms_per_step"]["median"] for e in entries) * 4.2)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("milliseconds per training step   (log scale, lower is better)",
                  fontsize=SZ_AXIS, color=theme["secondary"], labelpad=9)
    return fig


# --- Convergence --------------------------------------------------------------------

def smooth(values):
    window = max(2, min(len(values) // 12, 25))
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def equal_loss_threshold(traces):
    """A round loss every path crosses, about a third of the way down their shared range."""
    converging = [trace for trace in traces if not is_flat(trace)]
    floor = max(trace["trend"][-1] for trace in converging)
    ceiling = min(trace["trend"][0] for trace in converging)
    if ceiling <= floor:
        raise ValueError(f"[aether] Loss curves share no range: {ceiling} <= {floor}.")
    return round((floor + (ceiling - floor) * 0.35) * 20) / 20


def crossing_time(trace, threshold):
    return trace["t"][np.nonzero(trace["trend"] <= threshold)[0][0]]


def is_flat(trace):
    return trace["trend"][0] - trace["trend"][-1] < MIN_LOSS_DROP


def read_trace(result, label, linestyle="-"):
    loss = np.asarray(result["loss"])
    return {"key": result["key"], "label": label, "linestyle": linestyle, "loss": loss,
            "trend": smooth(loss), "t": np.asarray(result["elapsed_s"])}


def figure_convergence(paths, other_gpus, theme):
    results = {r["key"]: r for r in paths["results"]}
    main_traces = [read_trace(results[key], results[key]["label"]) for key in THROUGHPUT_ORDER]
    threshold = equal_loss_threshold(main_traces)
    main_times = [crossing_time(tr, threshold) for tr in main_traces if not is_flat(tr)]
    fastest, slowest = min(main_times), max(main_times)

    # Another GPU's fused path gets its own lime shade so it doesn't blur into the AMD green.
    other_traces = [{**read_trace(other["results"][0], f"{other['meta']['gpu']}, fused", DASHES),
                     "key": "other-gpu-fused"} for other in other_gpus]
    traces = main_traces + other_traces
    # Another GPU that never reaches the line still gets its curve, just no marker.
    crossings = sorted(((crossing_time(tr, threshold), tr) for tr in traces
                        if not is_flat(tr) and tr["trend"].min() <= threshold),
                       key=lambda item: item[0])

    fig = new_figure(
        theme,
        f"Same loss reached {fmt_multiple(slowest / fastest)} sooner on fused kernels",
        f"One epoch, {len(traces[0]['t'])} steps, same batch order and seed. Elapsed time "
        "with a sync every step, log scale.",
        footer_text(paths["meta"]),
    )
    ax = new_axes(fig, [0.092, 0.205, 0.865, 0.615], theme)
    ax.set_xscale("log")
    ax.grid(color=theme["grid"], linewidth=1.0)

    handles = []
    for trace in traces:
        color = theme["series"][trace["key"]]
        ax.plot(trace["t"], trace["loss"], color=color, linewidth=1.0, alpha=theme["raw_alpha"])
        handles.append(ax.plot(trace["t"], trace["trend"], color=color, linewidth=2.0,
                               linestyle=trace["linestyle"], label=f"{trace['label']}  ({fmt_time(trace['t'][-1])})")[0])
        ax.plot([trace["t"][-1]], [trace["trend"][-1]], marker="o", markersize=7.5, color=color,
                markeredgecolor=theme["surface"], markeredgewidth=2.0, zorder=5)
    ax.set_ylabel("training loss", fontsize=SZ_AXIS, color=theme["secondary"], labelpad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_xlim(min(tr["t"][0] for tr in traces) / 1.25, max(tr["t"][-1] for tr in traces) * 1.8)
    low, high = ax.get_ylim()
    # Headroom for the legend, which grows by a row per extra GPU, so it stays off the curves.
    ax.set_ylim(low - (high - low) * 0.14, high + (high - low) * (0.18 + 0.14 * len(other_traces)))
    low, high = ax.get_ylim()

    for trace in filter(is_flat, traces):
        ax.annotate("loss flat - did not converge", xy=(trace["t"][0], trace["trend"].mean()),
                    xytext=(-9, 0), textcoords="offset points", fontsize=SZ_AXIS,
                    color=theme["muted"], va="center", ha="right")

    ax.axhline(threshold, color=theme["axis"], linewidth=1.0, zorder=1)
    ax.annotate(f"loss {threshold:g}", xy=(0.006, threshold), xycoords=("axes fraction", "data"),
                xytext=(0, 6), textcoords="offset points", fontsize=SZ_AXIS,
                color=theme["secondary"])
    previous, tier = None, 0
    for t, trace in crossings:
        # Labels too close on the log axis to share a row step down onto the next.
        tier = (tier + 1) % 3 if previous and np.log10(t / previous) < 0.72 else 0
        previous = t
        label_y = low + (high - low) * (0.075 + 0.065 * tier)
        color = theme["series"][trace["key"]]
        ax.plot([t, t], [label_y, threshold], color=color, linewidth=1.0, alpha=0.55)
        ax.plot([t], [threshold], marker="o", markersize=7, color=color, zorder=6,
                markeredgecolor=theme["surface"], markeredgewidth=2.0)
        ax.annotate(fmt_time(t), xy=(t, label_y), xytext=(0, -3), textcoords="offset points",
                    fontsize=SZ_AXIS, color=theme["primary"], va="top", ha="center")

    legend = ax.legend(handles=handles[::-1], loc="upper right", framealpha=0.85,
                       facecolor=theme["legend_bg"], edgecolor=theme["axis"], fontsize=9.0,
                       handlelength=1.2, labelcolor=theme["primary"])
    legend.set_zorder(7)
    fig.text(0.5, 0.055,
             f"Equal loss ({threshold:g}) reached in {fmt_time(fastest)} vs {fmt_time(slowest)} "
             "elapsed",
             fontsize=SZ_CAPTION, color=theme["primary"], va="bottom", ha="center")
    return fig


# --- Transient VRAM -----------------------------------------------------------------

def figure_vram(vram, theme):
    results = {r["key"]: r for r in vram["results"]}
    peaks = {key: r["peak_transient_bytes"] for key, r in results.items()}
    floors = " · ".join(f"{key.split('-')[1]} {fmt_mib(r['baseline_bytes'])} MiB"
                             for key, r in results.items())
    gain = peaks["cupy-fallback"] / peaks["cupy-fused"]
    fig = new_figure(
        theme,
        f"Fused kernels hold {fmt_multiple(gain)} less transient VRAM",
        "Same CNN, same batches, same fp16 policy. Conv2d can only honor fp16 on the fused path.",
        footer_text(vram["meta"]),
    )
    fig.text(0.5, 0.092,
             "x-axis is position within the training step, not elapsed time - "
             "phase widths are fixed layout, not durations.",
             fontsize=SZ_CAPTION, color=theme["secondary"], va="bottom", ha="center")
    fig.text(0.5, 0.048,
             f"Transient only, above a resident floor of {floors} "
             "(parameters, AdamW state, batches)",
             fontsize=SZ_FOOTER, color=theme["muted"], va="bottom", ha="center")

    steps = vram["meta"]["recorded_steps"]
    ax = new_axes(fig, [0.092, 0.205, 0.865, 0.615], theme)
    ax.grid(axis="y", color=theme["grid"], linewidth=1.0)
    for key in VRAM_DRAW_ORDER:
        result = results[key]
        color = theme["series"][key]
        # A step function: the last value holds until the end of the last step.
        x = [*result["x"], float(steps)]
        y = [*result["transient_bytes"], result["transient_bytes"][-1]]
        ax.fill_between(x, 0.0, y, step="post", color=color, alpha=0.28, linewidth=0)
        ax.plot(x, y, color=color, linewidth=1.4, drawstyle="steps-post")

    # Zero-based, so the two filled areas stand in the ratio the title states.
    top = max(peaks.values()) * 1.34
    ax.set_xlim(0.0, steps)
    ax.set_ylim(0.0, top)
    tick = next(s for s in (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)
                if top / (s * MIB) <= 6.5) * MIB
    ax.set_yticks(np.arange(0.0, top, tick))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / MIB:,.0f}"))
    ax.set_ylabel("transient VRAM (MiB)", fontsize=SZ_AXIS, color=theme["secondary"], labelpad=6)
    ax.set_xticks([step + 0.5 for step in range(steps)],
                  [f"step {step + 1}" for step in range(steps)], color=theme["secondary"])

    position = 0.0
    for name, width in itertools.chain.from_iterable(
            itertools.repeat(tuple(vram["phase_widths"].items()), steps)):
        ax.annotate(name, xy=(position + width / 2, 0.985 - 0.062 * PHASE_LABEL_ROW[name]),
                    xycoords=("data", "axes fraction"), fontsize=SZ_FOOTER,
                    color=theme["secondary"], va="top", ha="center")
        position += width
        if position < steps:
            step_seam = abs(position - round(position)) < 1e-6
            ax.axvline(position, color=theme["axis"], linewidth=1.2 if step_seam else 1.0,
                       linestyle=(0, (2, 2)) if step_seam else (0, (1, 2)), zorder=1)

    for key in VRAM_DRAW_ORDER:
        result = results[key]
        x, peak = result["peak_x"], result["peak_transient_bytes"]
        color = theme["series"][key]
        ax.plot([x, x], [0.0, peak], color=color, linewidth=1.0, alpha=0.55)
        ax.plot([x], [peak], marker="o", markersize=7, color=color, zorder=6,
                markeredgecolor=theme["surface"], markeredgewidth=2.0)
        flip = x > steps * 0.55
        ax.annotate(f"{result['label']} \u00b7 {fmt_mib(peak)} MiB peak", xy=(x, peak), xytext=(-9 if flip else 9, 7),
                    textcoords="offset points", fontsize=SZ_AXIS, color=theme["primary"],
                    ha="right" if flip else "left", zorder=7,
                    bbox={"boxstyle": "round,pad=0.28", "facecolor": theme["surface"],
                          "edgecolor": "none", "alpha": 0.82})
    return fig


# --- Release comparison -------------------------------------------------------------

def stage_ms(breakdown):
    """GPU ms per step, grouped: forward, backward (layers only) and everything else."""
    stages = dict.fromkeys(STAGE_LABELS, 0.0)
    for row in breakdown["rows"]:
        stage = row["stage"] if row["stage"] in ("forward", "backward") else "rest"
        stages[stage] += row["gpu_ms"]["median"]
    return stages


def release_panel(ax, old_version, new_version, old, new, reference, gain, theme):
    # The remainder is small and carries no story, so it stays neutral.
    colors = {**theme["stages"], "rest": theme["muted"]}
    top = 2 if reference else 1
    ax.grid(axis="x", color=theme["grid"], linewidth=1.0)
    ax.spines["left"].set_visible(False)
    ax.set_title(new["meta"]["gpu"], loc="left", fontsize=SZ_CAPTION, fontweight="bold",
                 color=theme["primary"], pad=5)
    for row, breakdown in ((top, old), (top - 1, new)):
        left = 0.0
        for stage, ms in stage_ms(breakdown).items():
            ax.barh(row, ms, left=left, height=0.42, color=colors[stage],
                    edgecolor=theme["surface"], linewidth=1.0, label=STAGE_LABELS[stage])
            left += ms
        label = f"{fmt_ms(breakdown['baseline_ms']['median'])} ms/step"
        if row == top - 1:
            label += f"  ·  {fmt_multiple(gain)} faster"
        ax.annotate(label, xy=(left, row), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=SZ_AXIS, color=theme["primary"],
                    fontweight="bold" if row == top - 1 else "normal")
    ticks = [(top, old_version), (top - 1, new_version)]
    longest = sum(stage_ms(old).values())
    if reference:
        ms, new_ms = reference["baseline_ms"]["median"], new["baseline_ms"]["median"]
        ratio = ms / new_ms
        ax.barh(0, ms, height=0.42, color=theme["secondary"])
        ax.annotate(f"{fmt_ms(ms)} ms/step  ·  {new_version} is "
                    f"{fmt_multiple(ratio if ratio >= 1 else 1 / ratio)} "
                    f"{'faster' if ratio >= 1 else 'slower'}",
                    xy=(ms, 0), xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=SZ_AXIS, color=theme["primary"])
        ticks.append((0, "PyTorch"))
        longest = max(longest, ms)
    ax.set_yticks(*zip(*ticks), color=theme["primary"])
    ax.set_ylim(-0.5, top + 0.5)
    ax.set_xlim(0, longest * 1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} ms"))


def figure_release(old_version, new_version, runs, theme):
    gains = [old["baseline_ms"]["median"] / new["baseline_ms"]["median"] for old, new, _ in runs]
    headline = (f"{min(gains):.1f}–{max(gains):.1f}{TIMES}" if len(gains) > 1
                else fmt_multiple(gains[0]))
    first_old, first_new, _ = runs[0]
    reference = next((ref for _, _, ref in runs if ref), None)

    # Everything is laid out in inches, so the panels keep their size as GPUs are added.
    row_height, gap, top, bottom = 0.425, 0.62, 1.32, 0.60
    panels = [row_height * (3 if reference else 2) for _, _, reference in runs]
    height = top + bottom + sum(panels) + gap * (len(runs) - 1)
    fig = new_figure(
        theme,
        f"{new_version} trains {headline} faster than {old_version}",
        "Same CNN and fp16 policy. Segments are GPU time per stage, totals are ms/step."
        + (" Each GPU has its own scale." if len(runs) > 1 else ""),
        f"CIFAR-10 · batch {first_new['meta']['batch_size']} · seed {SEED} · "
        f"median of {first_new['meta']['repeats']} runs · "
        f"{first_old['meta']['commit'][:7]} vs {first_new['meta']['commit'][:7]}"
        + (f" · PyTorch {reference['meta']['torch'].split('+')[0]} compiled, fp16 autocast"
           if reference else ""),
        height=height,
    )
    for i, ((old, new, reference), gain) in enumerate(zip(runs, gains)):
        y = height - top - sum(panels[:i + 1]) - gap * i
        ax = new_axes(fig, [0.10, y / height, 0.86, panels[i] / height], theme)
        release_panel(ax, old_version, new_version, old, new, reference, gain, theme)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="upper left", bbox_to_anchor=(0.03, 1 - 0.76 / height),
               ncols=3, frameon=False, fontsize=SZ_FOOTER, handlelength=1.0,
               labelcolor=theme["secondary"])
    return fig


def load_release_runs(old_version, new_version):
    """(old, new, torch) for every GPU, fastest first; torch is None on a GPU without a
    torch_<gpu>.json. All runs must share one pair of commits, or the footer and the title
    would be describing different code."""
    runs = []
    for new_path in sorted((HISTORY / new_version).glob("breakdown_*.json")):
        old_path = HISTORY / old_version / new_path.name
        if not old_path.exists():
            raise FileNotFoundError(f"[aether] {old_path} is missing; {new_path} has no baseline.")
        torch_path = new_path.with_name(new_path.name.replace("breakdown_", "torch_", 1))
        runs.append((json.loads(old_path.read_text()), json.loads(new_path.read_text()),
                     json.loads(torch_path.read_text()) if torch_path.exists() else None))
    if not runs:
        raise FileNotFoundError(f"[aether] No breakdown_*.json in {HISTORY / new_version}.")
    commits = {(old["meta"]["commit"], new["meta"]["commit"]) for old, new, _ in runs}
    if len(commits) > 1:
        raise ValueError(f"[aether] Release runs measured different commits: {sorted(commits)}.")
    return sorted(runs, key=lambda run: run[1]["baseline_ms"]["median"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("old_version", help="folder in benchmarks/history, e.g. v1.1.0")
    parser.add_argument("new_version", help="folder in benchmarks/history, e.g. v1.2.0")
    args = parser.parse_args()

    results = HISTORY / args.new_version
    paths = json.loads((results / "paths.json").read_text())
    vram = json.loads((results / "vram.json").read_text())
    runs = load_release_runs(args.old_version, args.new_version)
    other_gpus = [json.loads(path.read_text())
                  for path in sorted(results.glob("paths_fused_*.json"))]

    for theme in THEMES.values():
        figures = {
            "benchmark_throughput": figure_throughput(paths, theme),
            "benchmark_convergence": figure_convergence(paths, other_gpus, theme),
            "benchmark_vram": figure_vram(vram, theme),
            "benchmark_release": figure_release(args.old_version, args.new_version, runs, theme),
        }
        for name, fig in figures.items():
            fig.savefig(ASSETS / f"{name}{theme['suffix']}.png", facecolor=theme["surface"])
            plt.close(fig)


if __name__ == "__main__":
    main()
