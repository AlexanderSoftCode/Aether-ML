import unittest
from unittest import mock

import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.custom_kernels import conv_kernel
from aether.custom_kernels.conv_kernel import common, dinput, dweight, forward


class TestChooseBlockTileN(unittest.TestCase):
    """32-channel GEMM N axes take a 32-wide block, wider ones stay at 64."""

    def test_tile_width(self):
        for n, expected in ((3, 32), (8, 32), (32, 32), (33, 64), (64, 64), (128, 64)):
            with self.subTest(n=n):
                self.assertEqual(conv_kernel.choose_block_tile_n(n), expected)

    def test_launch_tiles_follow_the_gemm_axes(self):
        # forward/dweight tile C_out, dinput tiles C_in; dweight's M axis is
        # K_TOTAL, which takes a 32-row tile up to 32 filter indices.
        meta = conv_kernel.ConvShapeMeta.create((8, 8, 3), 64, (3, 3), (1, 1), "same", is_hip=False)
        self.assertEqual(forward._forward_tiles(meta), (64, 64))
        self.assertEqual(dweight._dweight_tiles(meta), (32, 64))
        self.assertEqual(dinput._dinput_tiles(meta), (64, 32))
        meta = conv_kernel.ConvShapeMeta.create((8, 8, 64), 32, (3, 3), (1, 1), "same", is_hip=False)
        self.assertEqual(forward._forward_tiles(meta), (64, 32))
        self.assertEqual(dweight._dweight_tiles(meta), (64, 32))
        self.assertEqual(dinput._dinput_tiles(meta), (64, 64))
        self.assertEqual(common.threads_per_block(64, 32), 256)
        self.assertEqual(common.threads_per_block(32, 32), 128)

    def test_forward_reads_fp32_only_on_the_scalar_gather(self):
        for C_in, expected in ((3, np.float32), (12, np.float32), (8, np.float16), (64, np.float16)):
            with self.subTest(C_in=C_in):
                meta = conv_kernel.ConvShapeMeta.create((8, 8, C_in), 16, (3, 3), (1, 1), "same", is_hip=False)
                self.assertIs(conv_kernel.choose_forward_input_dtype(meta), expected)


class TestConvNarrowTileParity(base_case.AetherBaseTestCase):
    """The narrow tiles against the 64x64 tile on the same operands.

    Forward and dinput reduce every output in the same K order whatever the
    block shape, so they must agree bit for bit; dweight's split-K count
    follows the block size, so it is compared norm-relative.
    """

    # (H, W, C_in, C_out, filter, stride, padding): C = 8, 24, 32 on each axis,
    # plus a K_TOTAL <= 32 dweight (C_in = 3).
    CASES = (
        (9, 7, 8, 24, (3, 3), (1, 1), "same"),
        (8, 8, 24, 32, (3, 3), (2, 2), "same"),
        (10, 10, 32, 8, (4, 4), (2, 2), "same"),
        (9, 9, 3, 32, (3, 3), (1, 1), "valid"),
    )
    S = 3

    def setUp(self):
        super().setUp()
        if self.backend_name != "cupy":
            self.skipTest("CuPy backend required")
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")
        self.xp.random.seed(0)

    def _wide(self, plan, module, tiles_fn, render, name, grid, **render_kwargs):
        with mock.patch.object(module, tiles_fn, return_value=(64, 64)):
            source = render(plan.shape_meta, **render_kwargs)
        kernel = config.cp.RawModule(code=source, options=common._COMPILE_OPTIONS).get_function(name)
        return common.ConvLaunchPlan(
            kernel, grid, (common.threads_per_block(64, 64),), plan.M_total, plan.shape_meta
        )

    def test_narrow_tiles_match_the_64_wide_tile(self):
        xp, S = self.xp, self.S
        for H, W, C_in, C_out, filter_size, stride, padding in self.CASES:
            meta = conv_kernel.ConvShapeMeta.create((H, W, C_in), C_out, filter_size, stride, padding)
            x = xp.random.randn(S, H, W, C_in).astype(xp.float16)
            w = (xp.random.randn(meta.K_total, C_out) * 0.2).astype(xp.float16)
            bias = xp.random.randn(C_out).astype(xp.float32)
            dy = xp.random.randn(S, meta.H_out, meta.W_out, C_out).astype(xp.float16)
            out_shape = (S, meta.H_out, meta.W_out, C_out)
            n_tiles = lambda n: -(-n // 64)

            with self.subTest(kernel="forward", C_in=C_in, C_out=C_out):
                plan = conv_kernel.plan_forward_launch(meta, S, xp.float16)
                wide = self._wide(plan, forward, "_forward_tiles", forward.render_forward_source,
                                  "conv_forward_wmma", (plan.grid[0], n_tiles(C_out)),
                                  input_dtype=xp.float16)
                out, ref = xp.empty(out_shape, xp.float32), xp.empty(out_shape, xp.float32)
                plan(x, w, bias, out)
                wide(x, w, bias, ref)
                xp.testing.assert_array_equal(out, ref)

            with self.subTest(kernel="dinput", C_in=C_in, C_out=C_out):
                plan = conv_kernel.plan_dinput_launch(meta, S)
                wide = self._wide(plan, dinput, "_dinput_tiles", dinput.render_dinput_source,
                                  "conv_backward_dinput_wmma",
                                  (plan.grid[0], n_tiles(C_in), plan.grid[2]))
                out, ref = xp.empty(x.shape, xp.float32), xp.empty(x.shape, xp.float32)
                plan(dy, w, out)
                wide(dy, w, ref)
                xp.testing.assert_array_equal(out, ref)

            with self.subTest(kernel="dweight", C_in=C_in, C_out=C_out):
                plan = conv_kernel.plan_dweight_launch(meta, S)
                wide = self._wide(plan, dweight, "_dweight_tiles", dweight.render_dweight_source,
                                  "conv_backward_dweight_wmma",
                                  (n_tiles(meta.K_total), n_tiles(C_out), 1))
                results = []
                for launch in (plan, wide):
                    dw = xp.zeros((meta.K_total, C_out), xp.float32)
                    db = xp.zeros((C_out,), xp.float32)
                    launch(x, dy, dw, db)
                    results.append((dw, db))
                (dw, db), (dw_ref, db_ref) = results
                for label, actual, desired in (("dweights", dw, dw_ref), ("dbiases", db, db_ref)):
                    rel = float(xp.max(xp.abs(actual - desired))) / float(xp.max(xp.abs(desired)))
                    self.assertLessEqual(rel, 1e-5, f"{label}: relative error {rel:.3e}")


base_case.register_test_suites(globals(), template_cls=TestConvNarrowTileParity)
