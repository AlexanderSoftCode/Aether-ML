import aether.config as config
import tests.base_case as base_case
from aether.custom_kernels import conv_kernel
from aether.custom_kernels.conv_kernel import common, dinput, dweight, forward


class TestConvVectorStaging(base_case.AetherBaseTestCase):
    """The 16-byte staging paths against the scalar ones they replace.

    Each kernel is rendered a second time with vectorize=False, which forces
    the per-element staging, and launched on the same grid. Forward and dinput
    stage the same values into the same tiles either way, so they must agree
    bit for bit; dweight accumulates split-K slices with atomics, so it is
    compared norm-relative.
    """

    # (S, H, W, C_in, C_out, filter, stride, padding)
    CASES = (
        (2, 9, 7, 16, 24, (3, 3), (1, 1), "same"),    # both tiles vectorized
        (2, 8, 8, 8, 8, (2, 2), (2, 2), "same"),      # even filter, strided
        (2, 7, 9, 3, 16, (3, 3), (1, 1), "same"),     # C_in = 3: scalar A, vector B
        (2, 9, 9, 12, 20, (3, 3), (2, 2), "valid"),   # C % 8 != 0: scalar both
        (3, 10, 10, 32, 40, (4, 4), (2, 2), "same"),  # partial N tile
        (2, 6, 6, 24, 8, (1, 1), (1, 1), "valid"),    # 1x1 filter
    )

    def setUp(self):
        super().setUp()
        if self.backend_name != "cupy":
            self.skipTest("CuPy backend required")
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")
        self.xp.random.seed(0)

    def _scalar(self, plan, render, name):
        source = render(plan.shape_meta, vectorize=False)
        kernel = config.cp.RawModule(code=source, options=common._COMPILE_OPTIONS).get_function(name)
        return common.ConvLaunchPlan(kernel, plan.grid, plan.block, plan.M_total, plan.shape_meta)

    def _operands(self, case):
        S, H, W, C_in, C_out, filter_size, stride, padding = case
        xp = self.xp
        meta = conv_kernel.ConvShapeMeta.create((H, W, C_in), C_out, filter_size, stride, padding)
        x = xp.random.randn(S, H, W, C_in).astype(xp.float16)
        w = (xp.random.randn(meta.K_total, C_out) * 0.2).astype(xp.float16)
        dy = xp.random.randn(S, meta.H_out, meta.W_out, C_out).astype(xp.float16)
        return S, meta, x, w, dy

    def test_forward_matches_scalar_staging(self):
        xp = self.xp
        for case in self.CASES:
            with self.subTest(case=case):
                S, meta, x, w, _ = self._operands(case)
                bias = xp.random.randn(meta.C_out).astype(xp.float32)
                plan = conv_kernel.plan_forward_launch(meta, S)
                scalar = self._scalar(plan, forward.render_forward_source, "conv_forward_wmma")
                out_shape = (S, meta.H_out, meta.W_out, meta.C_out)
                out, ref = xp.empty(out_shape, xp.float32), xp.empty(out_shape, xp.float32)
                plan(x, w, bias, out)
                scalar(x, w, bias, ref)
                xp.testing.assert_array_equal(out, ref)

    def test_dinput_matches_scalar_staging(self):
        xp = self.xp
        for case in self.CASES:
            with self.subTest(case=case):
                S, meta, x, w, dy = self._operands(case)
                plan = conv_kernel.plan_dinput_launch(meta, S)
                scalar = self._scalar(plan, dinput.render_dinput_source, "conv_backward_dinput_wmma")
                out, ref = xp.empty(x.shape, xp.float32), xp.empty(x.shape, xp.float32)
                plan(dy, w, out)
                scalar(dy, w, ref)
                xp.testing.assert_array_equal(out, ref)

    def test_dweight_matches_scalar_staging(self):
        xp = self.xp
        for case in self.CASES:
            with self.subTest(case=case):
                S, meta, x, _, dy = self._operands(case)
                plan = conv_kernel.plan_dweight_launch(meta, S)
                scalar = self._scalar(plan, dweight.render_dweight_source, "conv_backward_dweight_wmma")
                results = []
                for launch in (plan, scalar):
                    dw = xp.zeros((meta.K_total, meta.C_out), xp.float32)
                    db = xp.zeros((meta.C_out,), xp.float32)
                    launch(x, dy, dw, db)
                    results.append((dw, db))
                (dw, db), (dw_ref, db_ref) = results
                for name, actual, desired in (("dweights", dw, dw_ref), ("dbiases", db, db_ref)):
                    rel = float(xp.max(xp.abs(actual - desired))) / float(xp.max(xp.abs(desired)))
                    self.assertLessEqual(rel, 1e-5, f"{name}: relative error {rel:.3e}")


base_case.register_test_suites(globals(), template_cls=TestConvVectorStaging)
