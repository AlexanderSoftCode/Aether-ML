import io
import re
import unittest
from unittest import mock

import numpy as np

from aether._utils.progress import TrainingProgress, _visible_len, make_progress
from aether.layers import Dense
from aether.losses import CategoricalCrossEntropy
from aether.metrics import CategoricalAccuracy
from aether.model import Model
from aether.optimizers import Adam

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_MAX_TRAILING = 16


def _overwrite(segments):
    """Applies carriage-return overwrite semantics to one rendered line."""
    buf = []
    for seg in segments:
        buf[0:len(seg)] = list(seg)
    return "".join(buf)


def _drive(progress, epochs=3, steps=40, log_every=8):
    """Runs a full multi-epoch lifecycle, capturing everything written."""
    sink = io.StringIO()
    progress._write = sink.write
    progress._flush = lambda: None

    losses = [1.5, 0.87, 1.4321, 0.9, 1.05]
    accs = [0.5, 0.8125, 0.6, 1.0, 0.0625]
    lrs = [9.62e-4, 1e-3, 5e-5, 1.234e-6, 8.1e-4]

    for epoch in range(1, epochs + 1):
        progress.start_epoch(epoch)
        for step in range(1, steps + 1):
            progress.tick(step, force=True)
            if step % log_every == 0 or step == steps:
                i = step % len(losses)
                progress.update_metrics(
                    step, losses[i], accs[i], lrs[i], reg_loss=0.0007 + i * 1e-4
                )
        progress.commit_epoch(epoch, 1.1, 0.5, lrs[epoch % len(lrs)])
        progress.commit_validation(0.9 - epoch * 0.05, 0.6 + epoch * 0.02)
    progress.close()
    return sink.getvalue()


class TestJupyterFrameCoverage(unittest.TestCase):
    """The padded frame must cover its predecessor without a visible band."""

    def setUp(self):
        self.stream = _drive(
            TrainingProgress(40, 3, has_reg=True, render_for="jupyter")
        )

    def _rendered_lines(self):
        for raw_line in self.stream.split("\n"):
            if "\r" in raw_line:
                yield raw_line

    def test_no_residue_under_either_overwrite_model(self):
        for raw_line in self._rendered_lines():
            for model, strip in (("raw-index", False), ("column", True)):
                segments = [
                    _ANSI.sub("", s) if strip else s for s in raw_line.split("\r")
                ]
                stranded = _overwrite(segments)[len(segments[-1]):]
                self.assertEqual(
                    stranded.strip(),
                    "",
                    f"{model}: previous frame left stranded: {stranded[:80]!r}",
                )

    def test_committed_lines_carry_no_whitespace_band(self):
        for raw_line in self._rendered_lines():
            visible = _ANSI.sub("", _overwrite(raw_line.split("\r")))
            trailing = len(visible) - len(visible.rstrip())
            self.assertLessEqual(
                trailing,
                _MAX_TRAILING,
                f"{trailing} trailing spaces on {visible.rstrip()[-40:]!r}",
            )

    def test_frozen_frame_keeps_its_telemetry(self):
        # Jupyter draws bar and telemetry as one line, so the epoch summary
        # lands beneath the frame rather than consuming it.
        frozen = [
            _ANSI.sub("", _overwrite(line.split("\r")))
            for line in self._rendered_lines()
        ]
        self.assertTrue(frozen, "no live frames were rendered")
        for line in frozen:
            self.assertIn("100%", line)
            self.assertIn("loss", line)
            self.assertIn("lr", line)

    def test_epoch_summary_follows_each_frozen_frame(self):
        summaries = [ln for ln in self.stream.split("\n") if "Total]" in ln]
        self.assertEqual(len(summaries), 3)


class TestOtherRenderModes(unittest.TestCase):
    """tty and plain paths must not inherit the jupyter padding scheme."""

    def test_tty_erases_with_ansi_rather_than_padding(self):
        stream = _drive(TrainingProgress(40, 2, has_reg=True, render_for="tty"))
        self.assertIn("\033[2K", stream)
        for raw_line in stream.split("\n"):
            visible = _ANSI.sub("", raw_line.split("\r")[-1])
            trailing = len(visible) - len(visible.rstrip())
            self.assertLessEqual(trailing, _MAX_TRAILING)

    def test_tty_live_lines_never_wrap_a_narrow_terminal(self):
        # The redraw moves the cursor up exactly one row per live line, so a
        # wrapped line would strand its overflow on screen.
        size = mock.Mock(columns=40)
        with mock.patch("shutil.get_terminal_size", return_value=size):
            stream = _drive(TrainingProgress(40, 2, has_reg=True, render_for="tty"))

        # Everything except the committed summary lines is a live frame.
        committed = ("Total]", "[Validation]", "[Summary]")
        live = [
            seg
            for seg in re.split(r"\r|\n|\033\[1A", stream)
            if seg and not any(tag in seg for tag in committed)
        ]
        self.assertTrue(live)
        for seg in live:
            self.assertLessEqual(_visible_len(seg), 39, _ANSI.sub("", seg))

    def test_plain_emits_no_escapes_and_no_carriage_returns(self):
        stream = _drive(TrainingProgress(40, 2, has_reg=True, render_for="plain"))
        self.assertNotIn("\r", stream)
        self.assertNotIn("\033", stream)
        self.assertIn("Epoch 1/2", stream)


class TestRunSummary(unittest.TestCase):
    """The closing `[Summary]` line is run-level: emitted once, only when
    there is something to report, and colorless off-terminal."""

    def _summaries(self, stream):
        return [ln for ln in stream.split("\n") if "[Summary]" in ln]

    def test_emitted_exactly_once_in_every_render_mode(self):
        for mode in ("tty", "jupyter", "plain"):
            with self.subTest(mode=mode):
                stream = _drive(TrainingProgress(40, 2, has_reg=True, render_for=mode))
                lines = self._summaries(stream)
                self.assertEqual(len(lines), 1, f"{mode}: {len(lines)} summary lines")

                visible = _ANSI.sub("", lines[0])
                self.assertIn("Total Time:", visible)
                self.assertIn("Best Val Acc:", visible)
                self.assertIn("Final Loss:", visible)

    def test_lands_on_its_own_line_after_a_blank_separator(self):
        # Both live modes leave the cursor on a fresh line, so the summary is
        # a standalone block -- never appended to a frozen frame.
        for mode in ("tty", "jupyter"):
            with self.subTest(mode=mode):
                stream = _drive(TrainingProgress(40, 2, has_reg=True, render_for=mode))
                lines = stream.split("\n")
                idx = next(i for i, ln in enumerate(lines) if "[Summary]" in ln)

                self.assertEqual(lines[idx - 1], "", f"{mode}: no blank separator")
                self.assertNotIn("\r", lines[idx])
                self.assertTrue(lines[idx].startswith("\033[96m[Summary]"))

    def test_reports_final_epoch_loss_and_best_val_acc(self):
        progress = TrainingProgress(4, 3, render_for="plain")
        sink = io.StringIO()
        progress._write = sink.write
        progress._flush = lambda: None

        for epoch, (loss, val_acc) in enumerate(
            ((1.9, 0.41), (1.55, 0.5296), (1.3477, 0.5012)), start=1
        ):
            progress.start_epoch(epoch)
            progress.commit_epoch(epoch, loss, 0.5, 1e-3)
            progress.commit_validation(0.9, val_acc)
        progress.close()

        line = self._summaries(sink.getvalue())[0]
        self.assertIn("Final Loss: 1.3477", line)
        # Best, not last: epoch 2 outscores the epoch 3 validation.
        self.assertIn("Best Val Acc: 52.96%", line)

    def test_accepts_already_scaled_validation_accuracy(self):
        # `commit_validation` normalizes 0-1 inputs to percent; values above 1
        # are already percentages and must not be scaled a second time.
        progress = TrainingProgress(4, 1, render_for="plain")
        sink = io.StringIO()
        progress._write = sink.write
        progress._flush = lambda: None

        progress.start_epoch(1)
        progress.commit_epoch(1, 1.0, 0.5, 1e-3)
        progress.commit_validation(0.9, 52.96)
        progress.close()

        self.assertIn("Best Val Acc: 52.96%", self._summaries(sink.getvalue())[0])

    def test_omits_val_acc_segment_without_validation(self):
        progress = TrainingProgress(4, 1, render_for="plain")
        sink = io.StringIO()
        progress._write = sink.write
        progress._flush = lambda: None

        progress.start_epoch(1)
        progress.commit_epoch(1, 1.0, 0.5, 1e-3)
        progress.close()

        line = self._summaries(sink.getvalue())[0]
        self.assertNotIn("Best Val Acc", line)
        self.assertIn("Total Time:", line)
        self.assertIn("Final Loss: 1.0000", line)

    def test_silent_when_no_epoch_committed(self):
        for mode in ("tty", "jupyter", "plain"):
            with self.subTest(mode=mode):
                progress = TrainingProgress(4, 1, render_for=mode)
                sink = io.StringIO()
                progress._write = sink.write
                progress._flush = lambda: None

                progress.close()
                self.assertEqual(self._summaries(sink.getvalue()), [])

    def test_repeated_close_does_not_duplicate(self):
        progress = TrainingProgress(4, 1, render_for="plain")
        sink = io.StringIO()
        progress._write = sink.write
        progress._flush = lambda: None

        progress.start_epoch(1)
        progress.commit_epoch(1, 1.0, 0.5, 1e-3)
        progress.close()
        progress.close()

        self.assertEqual(len(self._summaries(sink.getvalue())), 1)


class TestTrainInterrupt(unittest.TestCase):
    """An exception mid-epoch must still release the live block."""

    def test_interrupt_leaves_cursor_on_fresh_line(self):
        model = Model()
        model.to("numpy")
        model.add(Dense(n_inputs=4, n_neurons=3))
        model.configure(
            loss=CategoricalCrossEntropy(),
            optimizer=Adam(lr=0.01),
            accuracy=CategoricalAccuracy(),
        )
        model.finalize(input_shape=(4,))

        X = np.random.randn(32, 4).astype("float32")
        y = np.random.randint(0, 3, size=(32,)).astype("int32")

        progress = TrainingProgress(4, 1, render_for="tty")
        sink = io.StringIO()
        progress._write = sink.write
        progress._flush = lambda: None

        forward = model.forward
        calls = iter(range(3))

        def interrupting_forward(*args, **kwargs):
            if next(calls) == 2:
                raise KeyboardInterrupt
            return forward(*args, **kwargs)

        model.forward = interrupting_forward

        with (
            mock.patch("aether.model.make_progress", return_value=progress),
            self.assertRaises(KeyboardInterrupt),
        ):
            model.train(X, y, epochs=1, batch_size=8, print_every=1)

        self.assertTrue(
            _ANSI.sub("", sink.getvalue()).endswith("\n"),
            "traceback would print on the live line",
        )
        # Frozen where it stopped, not reported as a completed epoch.
        self.assertNotIn("100%", sink.getvalue())


class TestVisibleLen(unittest.TestCase):
    def test_discounts_sgr_runs(self):
        self.assertEqual(_visible_len("\033[92m▲1.25%\033[0m"), 6)
        self.assertEqual(_visible_len("plain"), 5)


class TestMakeProgress(unittest.TestCase):
    def test_verbose_zero_is_silent_null_object(self):
        progress = make_progress(0, 10, 1)
        self.assertEqual(type(progress).__name__, "_NullProgress")

    def test_verbose_two_forces_plain(self):
        self.assertEqual(make_progress(2, 10, 1).render_mode, "plain")

    def test_null_progress_parity(self):
        # The silent path calls the display unconditionally, so any method
        # added to TrainingProgress needs a counterpart on _NullProgress.
        null = make_progress(0, 10, 1)
        live = TrainingProgress(10, 1, render_for="plain")
        public = {
            name for name in vars(TrainingProgress)
            if not name.startswith("_") and callable(getattr(live, name))
        }
        missing = sorted(name for name in public if not hasattr(null, name))
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
