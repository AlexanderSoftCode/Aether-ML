"""
ANSI-based live training display supporting dual-line redraws, rolling
telemetry deltas, and scroll-to-history epoch/validation summaries.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Callable, Literal
from aether._utils.null_objects import _NullProgress

RenderMode = Literal["tty", "jupyter", "plain"]

# ~30 Hz. Above this, redraws are imperceptible and cost one write+flush
# syscall per step, coupling host overhead to model throughput.
_MIN_REDRAW_INTERVAL = 1.0 / 30.0


class Fore:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    RESET = "\033[39m"


class Style:
    DIM = "\033[2m"
    BRIGHT = "\033[1m"
    RESET_ALL = "\033[0m"


_GOODNESS = {
    "loss": -1,
    "acc": 1,
}

# Every possible bar fill (0-30 cells), built once at import.
_BAR_LOOKUP_COLOR = tuple(
    f"{Fore.YELLOW}{'█' * i}{Style.DIM}{'░' * (30 - i)}{Style.RESET_ALL}"
    for i in range(30)
) + (f"{Fore.GREEN}{'█' * 30}{Style.RESET_ALL}",)

_BAR_LOOKUP_PLAIN = tuple("█" * i + "░" * (30 - i) for i in range(31))

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _visible_len(text: str) -> int:
    """Printed column count of ``text``, ignoring embedded ANSI escapes."""
    return len(_ANSI_RE.sub("", text))


def _clip(text: str) -> str:
    """
    Truncates ``text`` to one terminal row. tty redraws move the cursor up a
    single row, so a live line that wraps would strand its overflow on screen.
    ANSI escapes are kept intact and cost no columns.
    """
    width = shutil.get_terminal_size().columns - 1
    if _visible_len(text) <= width:
        return text

    out, cols, pos = [], 0, 0
    for match in _ANSI_RE.finditer(text):
        chunk = text[pos:match.start()][: width - cols]
        out.append(chunk)
        out.append(match.group())
        cols += len(chunk)
        pos = match.end()
    out.append(text[pos:][: width - cols])
    return "".join(out) + Style.RESET_ALL


def _to_pct(acc: float) -> float:
    """Normalizes accuracy to percent; values above 1 are already percentages."""
    return acc * 100.0 if 0.0 <= acc <= 1.0 else acc


def _detect_render_mode(force_mode: str | None = None) -> RenderMode:
    """Detects whether standard cursor movement, CR overwrite, or plain text should be used."""
    if force_mode in ("tty", "jupyter", "plain"):
        return force_mode  # type: ignore[return-value]
    if "JPY_PARENT_PID" in os.environ or type(sys.stdout).__name__ == "OutStream":
        return "jupyter"
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return "tty"
    return "plain"


def _fmt_duration(seconds: float) -> str:
    """MM:SS past a minute, fractional seconds below it."""
    if seconds >= 60.0:
        total = int(seconds)
        return f"{total // 60:02d}:{total % 60:02d}"
    return f"{seconds:.1f}s"


def _delta_str(
    metric_name: str,
    current: float,
    previous: float | None,
    is_pct: bool = False,
    use_color: bool = True,
) -> str:
    """
    Renders delta glyphs (▲/▼) colored according to metric goodness:
    (goodness * direction > 0 -> Green, < 0 -> Red). Never fabricates 0.000.
    """
    if previous is None:
        return " " * 8

    diff = current - previous
    if abs(diff) < 1e-7:
        return " " * 8

    direction_sign = 1 if diff > 0 else -1
    goodness_sign = _GOODNESS.get(metric_name, 1) * direction_sign
    glyph = "▲" if diff > 0 else "▼"
    val_diff = abs(diff)

    if is_pct:
        formatted = f"{glyph}{val_diff:.2f}%"
    else:
        formatted = f"{glyph}{val_diff:.4f}"

    content = f"{formatted:>8}"

    if not use_color:
        return content

    color = Fore.GREEN if goodness_sign > 0 else Fore.RED
    return f"{color}{content}{Style.RESET_ALL}"


class TrainingProgress:
    """
    Host-side progress coordinator for training loops.

    Parameters
    ----------
    total_steps : int
        Total training batches/steps in a single epoch.
    epochs : int
        Total number of training epochs.
    has_reg : bool, default=False
        Whether the model utilizes regularization loss.
    render_for : str, optional
        Force a specific render mode ('tty', 'jupyter', or 'plain').
    """

    def __init__(
        self,
        total_steps: int,
        epochs: int,
        has_reg: bool = False,
        render_for: str | None = None,
    ) -> None:
        self.total_steps = max(total_steps, 1)
        self.epochs = epochs
        self.has_reg = has_reg
        self.render_mode: RenderMode = _detect_render_mode(render_for)
        self.use_color: bool = self.render_mode in ("tty", "jupyter")
        self._is_live: bool = self.render_mode != "plain"

        # Scale factors mapping a step onto bar cells and percent.
        self._step_factor: float = 30.0 / self.total_steps
        self._percent_factor: float = 100.0 / self.total_steps
        self._step_width: int = len(str(self.total_steps))
        self._bar_lookup = _BAR_LOOKUP_COLOR if self.use_color else _BAR_LOOKUP_PLAIN

        # Empty when color is off, so format strings need no branching.
        self._dim: str = Style.DIM if self.use_color else ""
        self._bright: str = Style.BRIGHT if self.use_color else ""
        self._reset: str = Style.RESET_ALL if self.use_color else ""

        # All output goes through these, so tests can swap in a sink.
        self._write: Callable[[str], int] = sys.stdout.write
        self._flush: Callable[[], None] = sys.stdout.flush

        # Render mode and reg-presence are fixed for the run, so pick each
        # implementation once.
        self._render: Callable[[], None] = {
            "tty": self._render_tty,
            "jupyter": self._render_jupyter,
        }.get(self.render_mode, self._render_noop)

        self._format_telemetry: Callable[..., str] = (
            self._format_telemetry_reg if has_reg else self._format_telemetry_noreg
        )

        # Run-level persistent state
        self._prev_val: dict[str, float | None] = {"loss": None, "acc": None}

        # Run summary state. The clock starts at construction -- `make_progress`
        # is called immediately before the epoch loop, so this spans the run.
        self._run_start_time: float = time.perf_counter()
        self._best_val_acc_pct: float | None = None
        self._final_train_loss: float | None = None
        self._closed: bool = False

        # Epoch-level state
        self._current_epoch: int = 1
        self._prev_train: dict[str, float | None] = {}
        self._epoch_start_time: float = 0.0

        # Redraw buffer state
        self._last_bar_line: str = ""
        self._last_telemetry_line: str = ""
        self._lines_on_screen: int = 0
        self._last_render_time: float = 0.0

        # Width of the jupyter frame currently on screen -- the only frame a
        # new one has to cover. Tracked in both units; see `_pad_jupyter`.
        self._live_jupyter_raw: int = 0
        self._live_jupyter_vis: int = 0

    # ---- Lifecycle -----------------------------------------------------

    def start_epoch(self, epoch: int) -> None:
        """Resets per-epoch intra-metric delta states and execution timer."""
        self._current_epoch = epoch
        self._prev_train.clear()
        self._epoch_start_time = time.perf_counter()
        self._last_bar_line = ""
        self._last_telemetry_line = ""
        self._lines_on_screen = 0
        self._last_render_time = 0.0
        self._live_jupyter_raw = 0
        self._live_jupyter_vis = 0

    def tick(self, step: int, force: bool = False) -> None:
        """
        Pure host-side progress bar update.
        """
        if not self._is_live:
            return

        now = time.perf_counter()

        if not (
            force
            or step >= self.total_steps
            or self._last_render_time == 0.0
            or (now - self._last_render_time) >= _MIN_REDRAW_INTERVAL
        ):
            return

        self._last_render_time = now
        self._last_bar_line = self._format_bar(step, now)
        self._render()

    def update_metrics(
        self,
        step: int,
        loss: float,
        acc: float,
        lr: float,
        reg_loss: float | None = None,
    ) -> None:
        """
        Called exclusively inside `print_every` gates.
        Computes metric deltas, updates telemetry, and executes a full redraw.
        """
        acc_pct = _to_pct(acc)

        loss_delta = _delta_str(
            "loss", loss, self._prev_train.get("loss"), is_pct=False, use_color=self.use_color
        )
        acc_delta = _delta_str(
            "acc", acc_pct, self._prev_train.get("acc"), is_pct=True, use_color=self.use_color
        )

        self._prev_train["loss"] = loss
        self._prev_train["acc"] = acc_pct

        self._last_telemetry_line = self._format_telemetry(
            loss, acc_pct, lr, loss_delta, acc_delta, reg_loss
        )

        # A metrics update is a real information change, so it bypasses the
        # redraw throttle.
        self.tick(step, force=True)

        if self.render_mode == "plain":
            self._write(
                f"Epoch {self._current_epoch}/{self.epochs} | "
                f"Step {step}/{self.total_steps} | {self._last_telemetry_line}\n"
            )
            self._flush()

    def commit_epoch(
        self,
        epoch: int,
        epoch_loss: float,
        epoch_acc: float,
        lr: float,
    ) -> None:
        """
        Freezes the completed bar into scrollback and writes the permanent
        epoch summary -- replacing the live telemetry line under ``tty``,
        landing beneath the combined live line under ``jupyter``.
        """
        self._final_train_loss = epoch_loss
        epoch_acc_pct = _to_pct(epoch_acc)

        dim, bright, reset = self._dim, self._bright, self._reset
        summary = (
            f"{bright}[Epoch {epoch}/{self.epochs} Total]{reset} "
            f"{dim}loss:{reset} {epoch_loss:.4f} - "
            f"{dim}acc:{reset} {epoch_acc_pct:.2f}% - "
            f"{dim}lr:{reset} {lr:.6f}"
        )

        self._freeze_live_block(
            self._format_bar(self.total_steps, time.perf_counter()), summary
        )

    def commit_validation(self, val_loss: float, val_acc: float) -> None:
        """Computes rolling epoch-over-epoch validation deltas and writes permanent summary."""
        val_acc_pct = _to_pct(val_acc)

        loss_delta = _delta_str(
            "loss", val_loss, self._prev_val.get("loss"), is_pct=False, use_color=self.use_color
        )
        acc_delta = _delta_str(
            "acc", val_acc_pct, self._prev_val.get("acc"), is_pct=True, use_color=self.use_color
        )

        if self._best_val_acc_pct is None or val_acc_pct > self._best_val_acc_pct:
            self._best_val_acc_pct = val_acc_pct
            badge = (
                f"   {Fore.YELLOW}★ new best acc{Style.RESET_ALL}"
                if self.use_color
                else "   ★ new best acc"
            )
        else:
            badge = ""

        self._prev_val["loss"] = val_loss
        self._prev_val["acc"] = val_acc_pct

        dim, bright, reset = self._dim, self._bright, self._reset
        self._write(
            f"{bright}[Validation]{reset} "
            f"{dim}loss:{reset} {val_loss:.4f} {loss_delta}   "
            f"{dim}acc:{reset} {val_acc_pct:.2f}% {acc_delta}{badge}\n"
        )
        self._flush()

    def close(self) -> None:
        """Freezes any dangling live block, emits the run summary, and resets
        terminal color state."""
        # Only reached with a live block on screen when training aborted
        # mid-epoch, so freeze the bar where it stopped rather than at 100%.
        self._freeze_live_block(self._last_bar_line)

        if not self._closed:
            self._closed = True
            summary = self._format_summary()
            if summary:
                self._write(f"\n{summary}\n")

        if self.use_color:
            self._write(Style.RESET_ALL)
        self._flush()

    def _format_summary(self) -> str | None:
        """
        One-shot run-level closing line. Returns ``None`` when no epoch ever
        committed -- a bare ``[Summary]`` with nothing to report is noise.
        """
        if self._final_train_loss is None:
            return None

        dim, reset = self._dim, self._reset
        cyan = Fore.CYAN if self.use_color else ""
        green = Fore.GREEN if self.use_color else ""
        yellow = Fore.YELLOW if self.use_color else ""

        elapsed = _fmt_duration(time.perf_counter() - self._run_start_time)
        parts = [f"{dim}Total Time:{reset} {elapsed}"]

        if self._best_val_acc_pct is not None:
            parts.append(
                f"{dim}Best Val Acc:{reset} "
                f"{green}{self._best_val_acc_pct:.2f}%{reset} {yellow}★{reset}"
            )

        parts.append(f"{dim}Final Loss:{reset} {self._final_train_loss:.4f}")

        return f"{cyan}[Summary]{reset} " + " - ".join(parts)

    # ---- Bar formatting ------------------------------------------

    def _format_bar(self, step: int, now: float) -> str:
        """
        Single formatter for both the live and frozen frames.
        """
        elapsed = now - self._epoch_start_time
        if step >= self.total_steps:
            bar_idx, pct = 30, 100
        else:
            bar_idx = min(30, max(0, int(step * self._step_factor)))
            pct = min(100, max(0, int(step * self._percent_factor)))

        ips = step / elapsed if (elapsed > 0.0 and step > 0) else 0.0

        return (
            f"[{self._bar_lookup[bar_idx]}] {pct:>3d}%  "
            f"Step {step:>{self._step_width}d}/{self.total_steps}  "
            f"{ips:.1f} it/s  {_fmt_duration(elapsed)}"
        )

    # ---- Telemetry formatting (variant chosen in __init__) ----------------

    def _format_telemetry_reg(
        self,
        loss: float,
        acc_pct: float,
        lr: float,
        loss_delta: str,
        acc_delta: str,
        reg_loss: float | None,
    ) -> str:
        dim, reset = self._dim, self._reset
        reg_str = f"{reg_loss:.4f}" if reg_loss is not None else "  --  "
        return (
            f"{dim}loss{reset} {loss:.4f}{loss_delta} "
            f"{dim}acc{reset} {acc_pct:.2f}%{acc_delta} "
            f"{dim}reg{reset} {reg_str}   "
            f"{dim}lr{reset} {lr:.2e}"
        )

    def _format_telemetry_noreg(
        self,
        loss: float,
        acc_pct: float,
        lr: float,
        loss_delta: str,
        acc_delta: str,
        reg_loss: float | None = None,
    ) -> str:
        dim, reset = self._dim, self._reset
        return (
            f"{dim}loss{reset} {loss:.4f}{loss_delta} "
            f"{dim}acc{reset} {acc_pct:.2f}%{acc_delta}   "
            f"{dim}lr{reset} {lr:.2e}"
        )

    # ---- Rendering --------------------------------------------

    def _render_tty(self) -> None:
        """Two-line in-place redraw. Cursor invariant: rests at end of last live line."""
        bar = _clip(self._last_bar_line)
        if self._last_telemetry_line:
            telemetry = _clip(self._last_telemetry_line)
            if self._lines_on_screen == 2:
                self._write(f"\033[1A\r\033[2K{bar}\n\033[2K{telemetry}")
            else:
                if self._lines_on_screen == 1:
                    self._write(f"\r\033[2K{bar}\n{telemetry}")
                else:
                    self._write(f"{bar}\n{telemetry}")
                self._lines_on_screen = 2
        else:
            if self._lines_on_screen == 1:
                self._write(f"\r\033[2K{bar}")
            elif self._lines_on_screen == 2:
                self._write(f"\033[1A\r\033[2K{bar}\n\033[2K")
                self._lines_on_screen = 1
            else:
                self._write(bar)
                self._lines_on_screen = 1
        self._flush()

    def _combine(self, bar_line: str) -> str:
        """Jupyter draws bar and telemetry as one line; tty keeps them apart."""
        return (
            f"{bar_line} | {self._last_telemetry_line}"
            if self._last_telemetry_line
            else bar_line
        )

    def _render_jupyter(self) -> None:
        self._write(f"\r{self._pad_jupyter(self._combine(self._last_bar_line))}")
        self._flush()
        self._lines_on_screen = 1

    def _pad_jupyter(self, line: str) -> str:
        """
        Right-pads a jupyter frame so it fully covers the frame it overwrites.
        """
        raw, vis = len(line), _visible_len(line)
        pad = max(self._live_jupyter_raw - raw, self._live_jupyter_vis - vis)

        if pad > 0:
            line = f"{line}{' ' * pad}"
            raw += pad
            vis += pad

        self._live_jupyter_raw = raw
        self._live_jupyter_vis = vis
        return line

    def _render_noop(self) -> None:
        """Plain mode: no live surface to maintain."""
        return

    # ---- Freeze helpers -----------------------------------

    def _freeze_live_block(self, bar: str, tail: str = "") -> None:
        """
        Commits the live block to scrollback with ``bar`` as its final frame,
        writes ``tail`` beneath it, and drops the cursor to a fresh line. Under
        ``tty`` an empty ``tail`` keeps the live telemetry line.
        """
        if self._lines_on_screen == 0:
            lines = [tail]
        elif self.render_mode == "tty":
            up = "\033[1A" if self._lines_on_screen == 2 else ""
            tail = tail or self._last_telemetry_line
            lines = [f"{up}\r\033[2K{_clip(bar)}", f"\033[2K{tail}" if tail else ""]
        else:  # jupyter -- single combined live line, frozen as drawn
            lines = [f"\r{self._pad_jupyter(self._combine(bar))}", tail]

        self._write("".join(f"{line}\n" for line in lines if line))
        self._flush()
        self._reset_live_state()

    def _reset_live_state(self) -> None:
        """
        Clears redraw bookkeeping so the next `_render` starts a fresh block
        rather than walking the cursor back into committed history.
        """
        self._lines_on_screen = 0
        self._last_bar_line = ""
        self._last_telemetry_line = ""
        self._last_render_time = 0.0
        self._live_jupyter_raw = 0
        self._live_jupyter_vis = 0


#: Shared stateless instance. `_NullProgress` holds no state, so one object
#: serves every silent run rather than allocating per `train()` call.
NULL_PROGRESS = _NullProgress()


def make_progress(
    verbose: int,
    total_steps: int,
    epochs: int,
    has_reg: bool = False,
    render_for: str | None = None,
) -> TrainingProgress | _NullProgress:
    """
    Maps a verbosity level onto a display object.

    Parameters
    ----------
    verbose : int
        ``0`` silent, ``1`` live bar with autodetected render mode,
        ``2`` forced plain text.
    total_steps : int
        Training batches in a single epoch.
    epochs : int
        Total number of training epochs.
    has_reg : bool, default=False
        Whether the model carries regularization loss.
    render_for : str, optional
        Escape hatch that pins the render mode outright.

    Returns
    -------
    TrainingProgress | _NullProgress
        Always call-compatible; the caller never branches on the result.
    """
    if verbose <= 0:
        return NULL_PROGRESS

    if render_for is None and verbose >= 2:
        render_for = "plain"

    return TrainingProgress(
        total_steps=total_steps,
        epochs=epochs,
        has_reg=has_reg,
        render_for=render_for,
    )