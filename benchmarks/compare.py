"""
Diff two benchmark result files and print only what changed.

Run:  python3 benchmarks/compare.py benchmarks/results/before.json benchmarks/results/after.json

The three summary lines are always shown. A stage, phase or bound forward path
is shown only when it changed; the rest is counted as hidden on the last line.

A phase changed when |after median - before median| is larger than the two runs'
min-max spreads added together, i.e. larger than the noise either run saw on its
own, and also larger than 2% and 0.01 ms. Stage sums have no spread, so only the
2% / 0.01 ms floor applies to them. Phases present in one file only (renamed,
fused or cut layers) are listed as added/removed.
"""
import argparse
import json
from pathlib import Path

CHECKED_META = ("gpu", "batch_size", "compute_dtype", "warmup_steps",
                "baseline_steps", "breakdown_steps")
MIN_CHANGE_PCT = 2.0
MIN_CHANGE_MS = 0.01


def width(stat):
    return stat["max"] - stat["min"]


def above_floor(old_ms, new_ms):
    """Whether a change exceeds MIN_CHANGE_PCT and MIN_CHANGE_MS, which 3 repeats
    can't resolve below."""
    return abs(new_ms - old_ms) > max(MIN_CHANGE_MS, MIN_CHANGE_PCT / 100 * old_ms)


def significant(old, new):
    """Whether two {median,min,max} stats differ by more than their combined spread
    and the floor."""
    return (abs(new["median"] - old["median"]) > width(old) + width(new)
            and above_floor(old["median"], new["median"]))


def fmt_line(label, label_width, old_ms, new_ms, sig=False):
    d = new_ms - old_ms
    pct = 100 * d / old_ms if old_ms else float("inf")
    return (f"{label:<{label_width}}  {old_ms:8.3f} {new_ms:8.3f}  "
            f"{d:+8.3f} {pct:+7.1f}% {'*' if sig else ' '}")


def describe(meta):
    commit = (meta["commit"] or "unknown")[:10]
    return f"{commit}{' (dirty)' if meta['dirty'] else ''}  {meta['date']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    old = json.loads(args.before.read_text())
    new = json.loads(args.after.read_text())

    print(f"before: {describe(old['meta'])}")
    print(f"after:  {describe(new['meta'])}")
    for key in CHECKED_META:
        if old["meta"][key] != new["meta"][key]:
            print(f"[aether] warning: {key} differs ({old['meta'][key]!r} vs "
                  f"{new['meta'][key]!r}); the numbers may not be comparable.")

    header_cols = f"{'before':>8} {'after':>8}  {'delta ms':>8} {'delta %':>8}"
    old_rows = {r["name"]: r for r in old["rows"]}
    new_rows = {r["name"]: r for r in new["rows"]}
    w = max(len(n) for n in old_rows | new_rows)

    print(f"\n{'summary':<{w}}  {header_cols}")
    for label, key in (("baseline model.train()", "baseline_ms"),
                       ("GPU total (instrumented)", "gpu_total_ms"),
                       ("host total (instrumented)", "host_total_ms")):
        o, n = old[key], new[key]
        print(fmt_line(label, w, o["median"], n["median"], significant(o, n)))

    stages = {}
    for side, rows in ((0, old_rows), (1, new_rows)):
        for r in rows.values():
            stages.setdefault(r["stage"], [0.0, 0.0])[side] += r["gpu_ms"]["median"]
    changed_stages = {s: v for s, v in stages.items() if above_floor(*v)}
    if changed_stages:
        print(f"\n{'GPU by stage':<{w}}  {header_cols}")
        for name, (o, n) in changed_stages.items():
            print(fmt_line(name, w, o, n))

    # (sort key, line): changed phases by |GPU delta|, then removed and added ones.
    lines = []
    unchanged = 0
    for name in (n for n in new_rows if n in old_rows):
        o, n = old_rows[name], new_rows[name]
        gpu_sig = significant(o["gpu_ms"], n["gpu_ms"])
        host_sig = significant(o["host_ms"], n["host_ms"])
        if not (gpu_sig or host_sig):
            unchanged += 1
            continue
        gpu_d = n["gpu_ms"]["median"] - o["gpu_ms"]["median"]
        host_d = n["host_ms"]["median"] - o["host_ms"]["median"]
        line = fmt_line(name, w, o["gpu_ms"]["median"], n["gpu_ms"]["median"], gpu_sig)
        lines.append((-abs(gpu_d), f"{line}  {host_d:+8.3f} {'*' if host_sig else ''}"))
    lines += [(0, f"{n:<{w}}  {old_rows[n]['gpu_ms']['median']:8.3f} {'':>8}  removed")
              for n in old_rows if n not in new_rows]
    lines += [(0, f"{n:<{w}}  {'':>8} {new_rows[n]['gpu_ms']['median']:8.3f}  added")
              for n in new_rows if n not in old_rows]
    if lines:
        print(f"\n{'GPU by phase':<{w}}  {header_cols}   host delta ms")
        for _, line in sorted(lines, key=lambda t: t[0]):
            print(line)

    old_paths, new_paths = old["paths"], new["paths"]
    changed_paths = [k for k in old_paths | new_paths if old_paths.get(k) != new_paths.get(k)]
    if changed_paths:
        print("\nbound forward path")
        for k in changed_paths:
            print(f"  {k}: {old_paths.get(k)} -> {new_paths.get(k)}")

    print(f"\nhidden as unchanged: {len(stages) - len(changed_stages)} stages, "
          f"{unchanged} phases, {len(old_paths | new_paths) - len(changed_paths)} paths. "
          f"* = beyond both runs' spread, {MIN_CHANGE_PCT:g}% and {MIN_CHANGE_MS:g} ms")


if __name__ == "__main__":
    main()
