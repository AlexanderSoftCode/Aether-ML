from unittest import mock

import aether.config as config
import tests.base_case as base_case
from aether.custom_kernels import conv_kernel


class TestConvKernelCompileFallback(base_case.AetherBaseTestCase):
    """GPU-side compile/load failures fall back to the scalar path; Python bugs surface."""

    # Odd shape so the key can't collide with one a real Conv2d already cached.
    SHAPE_META = conv_kernel.ConvShapeMeta(
        H_in=7, W_in=5, C_in=3, C_out=6, fH=3, fW=3, sH=1, sW=1,
        pad_h=0, pad_w=0, H_out=5, W_out=3, is_hip=config.IS_HIP,
    )
    # (getter, cache, key): forward also keys on its input dtype.
    GETTERS = (
        (conv_kernel.get_compiled_forward_conv_kernel, conv_kernel._CONV_FORWARD_KERNEL_CACHE,
         conv_kernel.forward_cache_key(SHAPE_META)),
        (conv_kernel.get_compiled_backward_dweight_kernel, conv_kernel._CONV_BACKWARD_DWEIGHT_CACHE,
         SHAPE_META.cache_key),
        (conv_kernel.get_compiled_backward_dinput_kernel, conv_kernel._CONV_BACKWARD_DINPUT_CACHE,
         SHAPE_META.cache_key),
    )

    def setUp(self):
        super().setUp()
        if self.backend_name != "cupy":
            self.skipTest("CuPy backend required")

    def _call_with_raw_module_raising(self, getter, cache, key, error):
        cache.pop(key, None)
        self.addCleanup(cache.pop, key, None)
        with mock.patch.object(config.cp, "RawModule", side_effect=error):
            return getter(self.SHAPE_META)

    def test_gpu_failure_warns_and_caches_none(self):
        errors = (
            config.cp.cuda.compiler.CompileException("bad source", "", "conv", ()),
            config.cp.cuda.driver.CUDADriverError(1),
        )
        for getter, cache, key in self.GETTERS:
            for error in errors:
                with self.subTest(getter=getter.__name__, error=type(error).__name__), \
                     self.assertWarns(UserWarning):
                    result = self._call_with_raw_module_raising(getter, cache, key, error)
                    self.assertIsNone(result)
                    self.assertIn(key, cache)
                    self.assertIsNone(cache[key])

    def test_cast_kernel_failure_warns_and_returns_none(self):
        cache, key = conv_kernel._CONV_BACKWARD_CAST_CACHE, config.IS_HIP
        if key in cache:
            self.addCleanup(cache.__setitem__, key, cache.pop(key))
        else:
            self.addCleanup(cache.pop, key, None)
        operand = self.xp.zeros((2, 3), dtype=self.xp.float32)
        error = config.cp.cuda.compiler.CompileException("bad source", "", "cast", ())
        with mock.patch.object(config.cp, "RawModule", side_effect=error), \
             self.assertWarns(UserWarning):
            self.assertIsNone(conv_kernel.cast_backward_operands_fp16(operand, operand))
        self.assertIsNone(cache[key])

    def test_python_error_is_not_swallowed(self):
        for getter, cache, key in self.GETTERS:
            with self.subTest(getter=getter.__name__):
                self.assertRaises(
                    TypeError, self._call_with_raw_module_raising, getter, cache, key, TypeError("bug")
                )
                self.assertNotIn(key, cache)


base_case.register_test_suites(globals(), template_cls=TestConvKernelCompileFallback)
