from unittest import mock

import tests.base_case as base_case
from aether.layers.conv import Conv2d
from tests.layers.conv2d.base import ConvLayerTestBase


class TestConvGpuParity(ConvLayerTestBase):
    """Matrix-core kernels against the fallback path (CuPy only; skipped on NumPy)."""

    def test_fp16_cache_invalidation_on_weight_update(self):
        """Ensure updating layer weights invalidates the FP16 shadow cache"""
        if self.backend_name == 'cupy':
            self.layer._compile_for_device('cupy')
            _ = self.layer.forward(self.test_images, training=False)
            
            # If GPU matrix-core path was taken, cache should now be valid
            if self.layer._fp16_weight_cache is not None:
                self.assertTrue(self.layer._fp16_weight_valid)

                # Update weights via property setter
                new_weights = self.xp.random.randn(*self.layer.filter_weights.shape).astype(self.xp.float32)
                self.layer.weights = new_weights
                self.assertFalse(self.layer._fp16_weight_valid)

    def test_forward_gpu_fallback_numeric_parity(self):
        """Verify numerical equivalence between _forward_gpu and _forward_fallback (CuPy backend only)"""
        if self.backend_name == 'cupy':
            from aether.custom_kernels import conv_kernel
            if conv_kernel.get_is_conv_gpu_available():
                fixed_input = self.xp.random.randn(2, 14, 14, self.IN_CHANNELS).astype(self.xp.float32)

                # Execute GPU matrix-core path
                gpu_out = self.layer._forward_gpu(fixed_input, training=False)

                # Execute CPU/CuPy fallback path
                fallback_out = self.layer._forward_fallback(fixed_input, training=False)

                # FP16 Tensor Cores introduce minor precision differences vs FP32 einsum
                self.xp.testing.assert_allclose(gpu_out, fallback_out, rtol=1e-2, atol=1e-2)

    def _run_dweight_kernel_parity(self, S, spatial, in_channels, out_channels,
                                   stride=None, padding=None, filter_size=None):
        """Drive the fused dweight kernel once and check both of its outputs.

        conv_backward_dweight_wmma emits dweights AND dbiases from a single
        launch -- the bias reduction rides along on the dY tile the GEMM has
        already staged. They are therefore checked together, off one launch,
        rather than as two independent tests.
        """
        if self.backend_name != 'cupy':
            self.skipTest("CuPy backend required for GPU kernel tests")
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")

        stride = stride or self.STRIDE
        padding = padding or self.PADDING
        filter_size = filter_size or self.FILTER_SIZE
        H, W = spatial

        layer = self.make_built_layer(
            Conv2d,
            input_shape=(H, W, in_channels),
            in_channels=in_channels,
            out_channels=out_channels,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
        )

        fixed_input = self.xp.random.randn(S, H, W, in_channels).astype(self.xp.float32)
        meta = layer._get_shape_meta((H, W, in_channels))
        dvalues = self.xp.random.randn(
            S, meta.H_out, meta.W_out, out_channels
        ).astype(self.xp.float32)

        layer._forward_fallback(fixed_input, training=True)
        layer._backward_fallback(dvalues)
        fallback_dweights = layer.dweights.copy()
        fallback_dbiases = layer.dbiases.copy()

        layer._forward_gpu(fixed_input, training=True)
        layer._backward_gpu(dvalues)
        gpu_dweights = layer.dweights.copy()
        gpu_dbiases = layer.dbiases.copy()

        self.assertEqual(gpu_dweights.shape, fallback_dweights.shape)
        self.assertEqual(gpu_dbiases.shape, fallback_dbiases.shape)

        self._assert_gemm_close(gpu_dweights, fallback_dweights, "dweights")
        self._assert_gemm_close(gpu_dbiases, fallback_dbiases, "dbiases")

        dvalues_fp16 = dvalues.astype(self.xp.float16).astype(self.xp.float32)
        ref_dbiases = self.xp.sum(dvalues_fp16, axis=(0, 1, 2))
        self.xp.testing.assert_allclose(gpu_dbiases, ref_dbiases, rtol=1e-5, atol=1e-5)

    def test_conv_gpu_backward_dweight_kernel_parity(self):
        """Fused dweight kernel: dweights and dbiases against the fallback."""
        self._run_dweight_kernel_parity(
            S=2, spatial=(8, 8), in_channels=self.IN_CHANNELS, out_channels=2
        )

    def test_conv_gpu_backward_dweight_kernel_parity_multichannel(self):
        """Same, on a shape whose K_TOTAL and C_OUT span several block tiles.

        The 8x8x1 -> 2 case above fits inside one 64x64 output tile, so it
        never exercises the grid.x/grid.y tiling or the ragged tail masking.
        """
        self._run_dweight_kernel_parity(
            S=4, spatial=(16, 16), in_channels=8, out_channels=48
        )

    def test_conv_gpu_backward_dweight_kernel_parity_valid_strided(self):
        """Same, with valid padding and a stride > 1."""
        self._run_dweight_kernel_parity(
            S=2, spatial=(16, 16), in_channels=4, out_channels=8,
            stride=(2, 2), padding="valid",
        )

    def _run_dinput_kernel_parity(self, S, spatial, in_channels, out_channels,
                                  stride=None, padding=None, filter_size=None):
        """Drive the gather dinput kernel and check it against the fallback."""
        if self.backend_name != 'cupy':
            self.skipTest("CuPy backend required for GPU kernel tests")
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")

        stride = stride or self.STRIDE
        padding = padding or self.PADDING
        filter_size = filter_size or self.FILTER_SIZE
        H, W = spatial

        layer = self.make_built_layer(
            Conv2d,
            input_shape=(H, W, in_channels),
            in_channels=in_channels,
            out_channels=out_channels,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            seed=self.SEED
        )

        fixed_input = self.xp.random.randn(S, H, W, in_channels).astype(self.xp.float32)
        meta = layer._get_shape_meta((H, W, in_channels))
        dvalues = self.xp.random.randn(
            S, meta.H_out, meta.W_out, out_channels
        ).astype(self.xp.float32)

        layer._forward_fallback(fixed_input, training=True)
        layer._backward_fallback(dvalues)
        fallback_dinputs = layer.dinputs.copy()

        layer._forward_gpu(fixed_input, training=True)
        layer._backward_gpu(dvalues)
        gpu_dinputs = layer.dinputs.copy()

        self.assertEqual(gpu_dinputs.shape, fallback_dinputs.shape)
        self._assert_gemm_close(gpu_dinputs, fallback_dinputs, "dinputs")

    def test_conv_gpu_backward_dinput_parity(self):
        """Baseline: 3x3 'same', stride 1 -- the shape the CIFAR model runs."""
        self._run_dinput_kernel_parity(
            S=2, spatial=(8, 8), in_channels=self.IN_CHANNELS, out_channels=2
        )

    def test_conv_gpu_backward_dinput_parity_multitile(self):
        """A shape whose M and C_IN both span several 64x64 block tiles.

        The 8x8 case above fits inside one output tile, so it never exercises
        the grid.x/grid.y tiling or the ragged tail masking in the epilogue.
        """
        self._run_dinput_kernel_parity(
            S=4, spatial=(16, 16), in_channels=8, out_channels=48
        )

    def test_conv_gpu_backward_dinput_parity_valid_strided(self):
        """stride 2 + valid padding: 4 residue classes, base_qh == 0."""
        self._run_dinput_kernel_parity(
            S=2, spatial=(16, 16), in_channels=4, out_channels=8,
            stride=(2, 2), padding="valid",
        )

    def test_conv_gpu_backward_dinput_parity_same_strided(self):
        """stride 2 + same padding: residue classes with a non-zero base_qh."""
        self._run_dinput_kernel_parity(
            S=2, spatial=(16, 16), in_channels=4, out_channels=8,
            stride=(2, 2), padding="same",
        )

    def test_conv_gpu_backward_dinput_parity_asymmetric(self):
        """Non-square filter with a mixed stride: sH != sW and fH != fW."""
        self._run_dinput_kernel_parity(
            S=2, spatial=(15, 12), in_channels=3, out_channels=6,
            filter_size=(5, 3), stride=(2, 1), padding="valid",
        )

    def test_conv_gpu_backward_dinput_matches_routed_fallback(self):
        """Assert both gpu and fallback paths agree on dinput tensor value"""
        if self.backend_name != "cupy":
            self.skipTest(
                "CuPy backend required: this test contrasts the matrix-core "
                "path against the fallback, so it needs both available"
            )
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")

        S, H, W, C_in, C_out = 2, 16, 16, 8, 16
        layer_kwargs = dict(
            input_shape=(H, W, C_in),
            seed=1234,
            in_channels=C_in,
            out_channels=C_out,
            filter_size=(3, 3),
            stride=(2, 2),
            padding="same",
        )

        gpu_layer = self.make_built_layer(Conv2d, **layer_kwargs)
        throwaway_layer = self.make_built_layer(Conv2d, **layer_kwargs)

        gpu_layer._compile_for_device("cupy")
        throwaway_layer._compile_for_device("numpy")

        self.assertIs(gpu_layer.forward.__func__, Conv2d._forward_gpu)
        self.assertIs(gpu_layer.backward.__func__, Conv2d._backward_gpu)
        self.assertIs(throwaway_layer.forward.__func__, Conv2d._forward_fallback)
        self.assertIs(throwaway_layer.backward.__func__, Conv2d._backward_fallback)

        throwaway_layer.set_parameters(
            weights=gpu_layer.filter_weights.copy(),
            biases=gpu_layer.biases.copy(),
        )

        inputs = self.xp.random.randn(S, H, W, C_in).astype(self.xp.float32)
        meta = gpu_layer._get_shape_meta((H, W, C_in))
        dvalues = self.xp.random.randn(
            S, meta.H_out, meta.W_out, C_out
        ).astype(self.xp.float32)

        gpu_layer.forward(inputs, training=True)
        gpu_dinputs = gpu_layer.backward(dvalues)

        throwaway_layer.forward(inputs, training=True)
        fallback_dinputs = throwaway_layer.backward(dvalues)

        self.assertEqual(gpu_dinputs.shape, (S, H, W, C_in))
        self.assertEqual(gpu_dinputs.shape, fallback_dinputs.shape)
        self._assert_gemm_close(
            gpu_dinputs, fallback_dinputs, "dinputs (routed gpu vs routed fallback)"
        )

    def test_conv_gpu_even_filter_forward_parity(self):
        """GPU forward reads the asymmetric pad off ConvShapeMeta like the fallback."""
        if self.backend_name != 'cupy':
            self.skipTest("CuPy backend required for GPU kernel tests")
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")
        for spatial, filter_size, stride in self.SAME_CASES:
            with self.subTest(spatial=spatial, filter_size=filter_size, stride=stride):
                layer = self._make_same_layer(spatial, filter_size, stride)
                x = self.xp.random.randn(2, *spatial, 2).astype(self.xp.float32)
                self._assert_gemm_close(
                    layer._forward_gpu(x, training=False),
                    layer._forward_fallback(x, training=False), "forward", tol=1e-2,
                )

    def test_conv_gpu_even_filter_dweight_parity(self):
        self._run_dweight_kernel_parity(
            S=2, spatial=(9, 9), in_channels=4, out_channels=8,
            stride=(2, 2), padding="same", filter_size=(4, 4),
        )

    def test_conv_gpu_even_filter_dinput_parity(self):
        self._run_dinput_kernel_parity(
            S=2, spatial=(8, 8), in_channels=4, out_channels=8,
            stride=(1, 1), padding="same", filter_size=(2, 2),
        )

    def test_conv_gpu_backward_parity_without_cast_kernel(self):
        """_backward_gpu's astype/clip path, taken when the fused cast kernel is unavailable."""
        from aether.custom_kernels import conv_kernel
        with mock.patch.object(conv_kernel, "cast_backward_operands_fp16", return_value=None) as cast:
            self._run_dweight_kernel_parity(S=2, spatial=(8, 8), in_channels=8, out_channels=20)
            self._run_dinput_kernel_parity(S=2, spatial=(8, 8), in_channels=8, out_channels=20)
        if self.backend_name == 'cupy' and conv_kernel.get_is_conv_gpu_available():
            self.assertGreater(cast.call_count, 0)

    def test_conv_gpu_partial_second_column_tile_parity(self):
        """84 = 64 + 16 + 4 channels on the GEMM N axis (C_out, or C_in for dinput).

        In the second 64-wide N tile warp 0 is full, warp 1 holds 4 valid
        columns and warps 2-3 are empty. The kernels skip empty warps, so a
        liveness check that ignores the tile offset or drops a partly valid
        warp loses columns here.
        """
        if self.backend_name != 'cupy':
            self.skipTest("CuPy backend required for GPU kernel tests")
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")

        layer = self._make_same_layer((8, 8), (3, 3), (1, 1), in_channels=8, out_channels=84)
        x = self.xp.random.randn(2, 8, 8, 8).astype(self.xp.float32)
        self._assert_gemm_close(
            layer._forward_gpu(x, training=False),
            layer._forward_fallback(x, training=False), "forward", tol=1e-2,
        )
        self._run_dweight_kernel_parity(S=2, spatial=(8, 8), in_channels=8, out_channels=84)
        self._run_dinput_kernel_parity(S=2, spatial=(8, 8), in_channels=84, out_channels=8)

    def test_conv_gpu_forward_view_inputs_match_contiguous(self):
        """Forward stages 16-byte vectors, so a view input is copied before launch.

        A 2-byte-offset fp16 slice breaks the alignment, a transposed view the
        contiguity; both must give the output of a fresh contiguous copy.
        """
        if self.backend_name != 'cupy':
            self.skipTest("CuPy backend required for GPU kernel tests")
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available on this device")

        xp = self.xp
        layer = self._make_same_layer((8, 8), (3, 3), (1, 1), in_channels=16, out_channels=16)
        x = xp.random.randn(2, 8, 8, 16).astype(xp.float16)

        buffer = xp.empty(x.size + 1, dtype=xp.float16)
        misaligned = buffer[1:].reshape(x.shape)
        misaligned[...] = x
        self.assertNotEqual(misaligned.data.ptr % 16, 0)
        transposed = xp.ascontiguousarray(x.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        self.assertFalse(transposed.flags.c_contiguous)

        expected = layer._forward_gpu(x, training=False).copy()
        for name, view in (("misaligned", misaligned), ("transposed", transposed)):
            with self.subTest(view=name):
                xp.testing.assert_array_equal(layer._forward_gpu(view, training=False), expected)

    def test_backward_fallback_on_forward_compile_failure(self):
        """Regression: when forward GPU kernel fails, backward must also fall back.

        When conv_kernel.plan_forward_launch returns None (JIT failure), _forward_gpu
        rebinds self.forward to the fallback. This test ensures self.backward is also
        rebound to the fallback, so backward doesn't raise on the first training step.
        """
        if self.backend_name != 'cupy':
            self.skipTest("CuPy backend required")
        from aether.custom_kernels import conv_kernel
        if not conv_kernel.get_is_conv_gpu_available():
            self.skipTest("Matrix-core GPU kernels not available")

        layer = self.make_built_layer(
            Conv2d,
            input_shape=(8, 8, self.IN_CHANNELS),
            seed=self.SEED,
            in_channels=self.IN_CHANNELS,
            out_channels=2,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING,
        )
        layer._compile_for_device('cupy')

        inputs = self.xp.random.randn(2, 8, 8, self.IN_CHANNELS).astype(self.xp.float32)
        meta = layer._get_shape_meta((8, 8, self.IN_CHANNELS))
        dvalues = self.xp.random.randn(
            2, meta.H_out, meta.W_out, 2
        ).astype(self.xp.float32)

        with mock.patch.object(conv_kernel, 'plan_forward_launch', return_value=None), \
             mock.patch.object(conv_kernel, 'plan_dweight_launch', return_value=None), \
             mock.patch.object(conv_kernel, 'plan_dinput_launch', return_value=None):
            # Forward should fall back without raising
            output = layer.forward(inputs, training=True)
            self.assertEqual(output.shape, (2, 8, 8, 2))

            # Backward should use the fallback binding, not raise
            dinputs = layer.backward(dvalues)

        # Assert backward is now bound to the fallback
        self.assertIs(layer.backward.__func__, Conv2d._backward_fallback)
        # Assert dinputs has the input shape
        self.assertEqual(dinputs.shape, inputs.shape)


base_case.register_test_suites(globals(), TestConvGpuParity)
