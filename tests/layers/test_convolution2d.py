import warnings
from unittest import mock

import numpy as np

import aether.config as config
import tests.base_case as base_case

from aether.layers.conv import Conv2d


class TestConvLayer(base_case.AetherBaseLayerTestCase):
    SEED = 42
    CONV_SHAPE = (2, 28, 28, 1)
    IN_CHANNELS=1
    OUT_CHANNELS=4
    FILTER_SIZE=(3,3)
    STRIDE=(1,1)
    PADDING="same"
    def setUp(self):
        super().setUp()
        self.as_strided = config.get_stride_utility(self.xp)
        
        self.layer = self.make_built_layer(
            Conv2d, 
            input_shape= self.CONV_SHAPE[1:],
            seed=self.SEED,
            in_channels=self.IN_CHANNELS, 
            out_channels=self.OUT_CHANNELS, 
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        # Seed the global stream too, not just the layer init. Test data is
        # drawn from it, so without this each test sees whatever stream
        # position the preceding tests left -- which makes a failure reproduce
        # under `unittest discover` but not when the file is run alone.
        self.xp.random.seed(self.SEED)

        #Dummy image, batch of 2, 28x28 pixels, 1 channel
        self.test_images = self.xp.random.randn(*self.CONV_SHAPE)

    def test_conv_forward_shape(self):
        """Verify output dimensions based on padding and stride"""
        output = self.layer.forward(self.test_images, training=True)
        # Given padding = 1, kernel = 3, 28x28 remainds 28x28 (same)
        self.assertEqual(output.shape, (2, 28, 28, 4))
    
    def test_parameters_initialization(self):
        """Compare filter weights and filter bias shapes"""
        self.assertEqual(self.layer.filter_weights.shape, (3, 3, 1, 4))
        self.assertEqual(self.layer.biases.shape, (4,))

    def test_backward_gradient_shapes(self):
        """Verify backprop gradients match weight shapes"""
        output = self.layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(*output.shape) #mock gradient
        self.layer.backward(dvalues)

        self.assertEqual(self.layer.dweights.shape, self.layer.filter_weights.shape)
        self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)

    def test_conv_same_padding_stride(self):
        # 28x28 input, 3x3 filter, stride=(2,2), padding='same'
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS, 
            out_channels=self.OUT_CHANNELS, 
            filter_size=self.FILTER_SIZE,
            stride=(2,2),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training=True)
        
        # Calculation: P=1 -> int((28 + 2 - 3)/2 + 1) = 14
        self.assertEqual(output.shape, (2, 14, 14, 4))
        
    def test_conv_valid_padding_stride(self):
        # 28x28 input, 3x3 filter, stride = (2,2), valid padding
        layer = self.make_built_layer(
            Conv2d, 
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS, 
            out_channels=self.OUT_CHANNELS, 
            filter_size=self.FILTER_SIZE,
            stride=(2,2),
            padding="valid"
        )
        output = layer.forward(self.test_images, training=True)
        # (28 - 3) / 2 + 1 = 13.5 -> floor to 13
        self.assertEqual(output.shape, (2, 13, 13, 4))

    def test_conv_zero_input(self):
        "Given an input of all zeros, show an output of all zeros (plus biases)"

        zero_input = self.xp.zeros((1, 28, 28, 1))
        
        # Expected: Every pixel in the output should equal the bias for that filter
        # We reshape biases to (1, 1, 1, num_filters) to match output shape
        
        output = self.layer.forward(zero_input, training = False)
        expected_output = self.xp.broadcast_to(self.layer.biases, output.shape)
        
        self.xp.testing.assert_array_almost_equal(output, expected_output)

    def test_conv_stride_length(self):

        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=(2,1),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training = False)
        
        self.assertEqual(output.shape, (2, 14, 28, 4))

    def test_conv_stride_width(self):
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=(1,2),
            padding=self.PADDING
        )
        output = layer.forward(self.test_images, training=False)
        self.assertEqual(output.shape, (2, 28, 14, 4))

    def test_conv_multichannel_input(self):
        """Verify forward pass with multi-channel input (e.g., RGB with C_in=3)"""
        layer = self.make_built_layer(
            Conv2d,
            input_shape = (28, 28, 3),
            in_channels=3,
            out_channels=8,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        rgb_images = self.xp.random.randn(2, 28, 28, 3)
        output = layer.forward(rgb_images, training=False)
        self.assertEqual(output.shape, (2, 28, 28, 8))

    def test_conv_variable_batch_sizes(self):
        """Verify shape cache handles dynamic batch sizes without re-compilation or shape errors"""
        inputs_b4 = self.xp.random.randn(4, 16, 16, self.IN_CHANNELS)
        out_b4 = self.layer.forward(inputs_b4, training=False)
        self.assertEqual(out_b4.shape, (4, 16, 16, self.OUT_CHANNELS))

        inputs_b1 = self.xp.random.randn(1, 16, 16, self.IN_CHANNELS)
        out_b1 = self.layer.forward(inputs_b1, training=False)
        self.assertEqual(out_b1.shape, (1, 16, 16, self.OUT_CHANNELS))

    def test_conv_non_square_filter(self):
        """Verify rectangular filter geometries such as (5, 3)"""
        layer = self.make_built_layer(
            Conv2d,
            input_shape = self.CONV_SHAPE[1:],
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=(5, 3),
            stride=(1, 1),
            padding="same"
        )
        output = layer.forward(self.test_images, training=False)
        self.assertEqual(output.shape, (2, 28, 28, self.OUT_CHANNELS))

    # ---- Conv.build() Seed Tests ----------

    def test_same_seed_produces_identical_weights(self):
        """Two independently instantiated layers built with the exact same seed
        must initialize with identical weight and bias values."""
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )

        seed_val = 123
        layer_1.build(input_shape = self.CONV_SHAPE[1:], seed=seed_val)
        layer_2.build(input_shape = self.CONV_SHAPE[1:], seed=seed_val)

        # Weights and biases must match exactly bit-for-bit on the active backend
        self.xp.testing.assert_array_equal(
            layer_1.weights, layer_2.weights,
            err_msg="Layers built with identical seeds produced diverging weights."
        )
        self.xp.testing.assert_array_equal(
            layer_1.biases, layer_2.biases,
            err_msg="Layers built with identical seeds produced diverging biases."
        )

    def test_different_seeds_produce_divergent_weights(self):
        """Two layers built with different seeds must produce different parameter initializations."""
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )

        layer_1.build(input_shape=self.CONV_SHAPE[1:], seed=123)
        layer_2.build(input_shape=self.CONV_SHAPE[1:], seed=456)

        # Weights should not be identical
        self.assertFalse(
            self.xp.array_equal(layer_1.weights, layer_2.weights),
            msg="Layers built with different seeds produced identical weights."
        )

    def test_build_seed_does_not_corrupt_global_rng_stream(self):
        """Calling build(seed=...) must draw from a local PRNG and avoid resetting
        or corrupting the backend's ambient global random state."""
        xp = self.xp

        # Seed global backend state and sample a reference number
        xp.random.seed(999)
        val_before = float(xp.random.randn(1)[0])

        # Reset global state to 999, build an intermediate layer with an isolated seed,
        # and verify the next global draw is unaffected
        xp.random.seed(999)
        throwaway_layer = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        throwaway_layer.build(input_shape = self.CONV_SHAPE[1:], seed=123)

        val_after = float(xp.random.randn(1)[0])

        self.assertAlmostEqual(
            val_before, val_after, places=6,
            msg="layer.build(seed=...) modified or reset the global backend random state."
        )

    def test_build_with_none_seed_uses_global_stream(self):
        """Calling build(seed=None) should sequentially consume random numbers
        from the global stream rather than throwing an error or using a static seed."""
        xp = self.xp

        xp.random.seed(777)
        layer_1 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_1.build(input_shape = self.CONV_SHAPE[1:], seed=None)

        xp.random.seed(777)
        layer_2 = Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
        layer_2.build(input_shape = self.CONV_SHAPE[1:], seed=None)

        self.xp.testing.assert_array_equal(
            layer_1.weights,
            layer_2.weights,
            err_msg="build(seed=None) failed to follow global backend seeding."
        )

    # ---- Gradient Check -----------------
    def test_conv_numerical_gradient_check(self):
        """
        Finite difference check: verify analytical dweights matches numerical approximation
        for the Conv. Uses a small input/filter to keep the double loop feasible.
        f'(x) ≈ (f(x + h) - f(x - h)) / 2h
        """
        epsilon = 1e-2
        layer = self.make_built_layer(
            Conv2d,
            input_shape = (8, 8, 1),
            in_channels=self.IN_CHANNELS,
            out_channels=2,
            filter_size=self.FILTER_SIZE,
            stride=self.STRIDE,
            padding=self.PADDING
        )
    
        fixed_input = self.xp.random.randn(2, 8, 8, 1)  # batch=2, 8x8, 1 channel
        dvalues = self.xp.ones((2, 8, 8, 2))             # ones so sum(output) is the scalar loss
    
        # Analytical gradient
        layer.forward(fixed_input, training=False)
        layer.backward(dvalues)
        analytical_dweights = layer.dweights.copy()
    
        # Numerical gradient — iterate over every element of filter_weights (3x3x1x2 = 18)
        numerical_dweights = self.xp.zeros_like(layer.filter_weights)
    
        for fh in range(layer.filter_weights.shape[0]):
            for fw in range(layer.filter_weights.shape[1]):
                for c in range(layer.filter_weights.shape[2]):
                    for f in range(layer.filter_weights.shape[3]):
    
                        # f(x + h) -- mutate the buffer in place, then invalidate,
                        # exactly like a fused optimizer kernel writing into
                        # layer.weights and calling invalidate_shadow_caches().
                        layer.filter_weights[fh, fw, c, f] += epsilon
                        layer.invalidate_shadow_caches()
                        layer.forward(fixed_input, training=False)
                        loss_plus = self.xp.sum(layer.output)
    
                        # f(x - h)
                        layer.filter_weights[fh, fw, c, f] -= 2 * epsilon
                        layer.invalidate_shadow_caches()
                        layer.forward(fixed_input, training=False)
                        loss_minus = self.xp.sum(layer.output)
    
                        # Restore -- also needs invalidating, so the layer is
                        # left in a correct state for the next element's pass
                        # (and for anything run after this loop).
                        layer.filter_weights[fh, fw, c, f] += epsilon
                        layer.invalidate_shadow_caches()
    
                        numerical_dweights[fh, fw, c, f] = (loss_plus - loss_minus) / (2 * epsilon)
    
        self.xp.testing.assert_allclose(
            analytical_dweights, 
            numerical_dweights, 
            rtol=3e-2,   # 3% relative tolerance
            atol=2e-2    # 0.02 absolute tolerance
        )
    # --- regression tests ---------------------------------------------------

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

    GEMM_REL_TOL = 5e-3

    def _to_numpy(self, arr):
        """Host copy, whichever backend the array came from."""
        return arr.get() if hasattr(arr, "get") else np.asarray(arr)

    def _assert_gemm_close(self, actual, desired, name, tol=None):
        """Norm-relative comparison: max|actual - desired| / max|desired|."""
        tol = self.GEMM_REL_TOL if tol is None else tol
        scale = float(self.xp.max(self.xp.abs(desired)))
        if scale == 0.0:
            scale = 1.0
        rel = float(self.xp.max(self.xp.abs(actual - desired))) / scale
        self.assertLessEqual(
            rel, tol,
            f"{name}: relative error {rel:.3e} exceeds {tol:.1e} "
            f"(max|ref|={scale:.3e})",
        )

    def _run_dweight_kernel_parity(self, S, spatial, in_channels, out_channels,
                                   stride=None, padding=None):
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
        H, W = spatial

        layer = self.make_built_layer(
            Conv2d,
            input_shape=(H, W, in_channels),
            in_channels=in_channels,
            out_channels=out_channels,
            filter_size=self.FILTER_SIZE,
            stride=stride,
            padding=padding,
        )

        fixed_input = self.xp.random.randn(S, H, W, in_channels).astype(self.xp.float32)
        meta = layer._get_shape_meta((H, W, in_channels))
        dvalues = self.xp.random.randn(
            S, meta.H_out, meta.W_out, out_channels
        ).astype(self.xp.float32)

        layer._forward_fallback(fixed_input, training=False)
        layer._backward_fallback(dvalues)
        fallback_dweights = layer.dweights.copy()
        fallback_dbiases = layer.dbiases.copy()

        layer._forward_gpu(fixed_input, training=False)
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

    # ---- dinput kernel regression tests ---------------------------

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

        layer._forward_fallback(fixed_input, training=False)
        layer._backward_fallback(dvalues)
        fallback_dinputs = layer.dinputs.copy()

        layer._forward_gpu(fixed_input, training=False)
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

        gpu_layer.forward(inputs, training=False)
        gpu_dinputs = gpu_layer.backward(dvalues)

        throwaway_layer.forward(inputs, training=False)
        fallback_dinputs = throwaway_layer.backward(dvalues)

        self.assertEqual(gpu_dinputs.shape, (S, H, W, C_in))
        self.assertEqual(gpu_dinputs.shape, fallback_dinputs.shape)
        self._assert_gemm_close(
            gpu_dinputs, fallback_dinputs, "dinputs (routed gpu vs routed fallback)"
        )


    @staticmethod
    def _brute_force_dinputs(x, weights, dvalues, stride, pad):
        """Reference dinputs from the definition of the convolution.

        Deliberately a naive scatter loop in fp32 NumPy: no strided views, no
        dilation trick, no im2col. Both production paths derive their geometry
        from the same reasoning, so neither can serve as an independent check
        on the other -- a shared misunderstanding of the output extents shows
        up as agreement, not as a failure. This owes them nothing.
        """
        S, H, W, _ = x.shape
        fH, fW, C_in, C_out = weights.shape
        sH, sW = stride
        pad_h, pad_w = pad
        _, H_out, W_out, _ = dvalues.shape

        dx = np.zeros((S, H, W, C_in), dtype=np.float32)
        for s in range(S):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    for fh in range(fH):
                        h_in = h_out * sH - pad_h + fh
                        if not (0 <= h_in < H):
                            continue
                        for fw in range(fW):
                            w_in = w_out * sW - pad_w + fw
                            if not (0 <= w_in < W):
                                continue
                            dx[s, h_in, w_in, :] += (
                                weights[fh, fw, :, :] @ dvalues[s, h_out, w_out, :]
                            )
        return dx

    def _run_dinput_ground_truth(self, S, spatial, in_channels, out_channels,
                                 stride, padding, filter_size=(3, 3)):
        """Check BOTH paths against the brute-force reference.

        The parity tests above pin the GPU kernel to the fallback, which is
        only as good as the fallback. A stride > 1 bug in the fallback's
        dilated-dY padding once read past its own buffer here and went
        unnoticed for exactly that reason: it returned zeros on a fresh
        allocator, which is the right answer for the rows it mis-covered, so
        the suite passed in isolation and failed only under discovery.
        """
        H, W = spatial
        layer = self.make_built_layer(
            Conv2d,
            input_shape=(H, W, in_channels),
            seed=self.SEED,
            in_channels=in_channels,
            out_channels=out_channels,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
        )
        meta = layer._get_shape_meta((H, W, in_channels))

        inputs = self.xp.random.randn(S, H, W, in_channels).astype(self.xp.float32)
        dvalues = self.xp.random.randn(
            S, meta.H_out, meta.W_out, out_channels
        ).astype(self.xp.float32)

        truth = self._brute_force_dinputs(
            self._to_numpy(inputs), self._to_numpy(layer.filter_weights),
            self._to_numpy(dvalues), stride, (meta.pad_h, meta.pad_w),
        )
        truth = self.xp.asarray(truth)

        layer._forward_fallback(inputs, training=False)
        layer._backward_fallback(dvalues)
        self._assert_gemm_close(layer.dinputs, truth, "dinputs (fallback vs truth)",
                                tol=1e-5)

        from aether.custom_kernels import conv_kernel
        if self.backend_name == 'cupy' and conv_kernel.get_is_conv_gpu_available():
            layer._forward_gpu(inputs, training=False)
            layer._backward_gpu(dvalues)
            self._assert_gemm_close(layer.dinputs, truth, "dinputs (gpu vs truth)")

    def test_conv_dinput_ground_truth_stride1(self):
        """Stride 1: the only case where the fallback's symmetric pad was right."""
        self._run_dinput_ground_truth(
            S=2, spatial=(16, 16), in_channels=4, out_channels=8,
            stride=(1, 1), padding="same",
        )

    def test_conv_dinput_ground_truth_strided_same(self):
        """Stride 2 'same': dilated_H falls 1 short of the H_in windows needed."""
        self._run_dinput_ground_truth(
            S=2, spatial=(16, 16), in_channels=4, out_channels=8,
            stride=(2, 2), padding="same",
        )

    def test_conv_dinput_ground_truth_strided_valid(self):
        """Stride 2 'valid': same shortfall, reached by a different pad rule."""
        self._run_dinput_ground_truth(
            S=2, spatial=(16, 16), in_channels=4, out_channels=8,
            stride=(2, 2), padding="valid",
        )

    def test_conv_dinput_ground_truth_mixed_stride(self):
        """sH != sW: height comes up short while width lands exactly in bounds.

        This asymmetry is what made the original bug look intermittent -- a
        shape whose width happens to fit reads in bounds on one axis and out
        of bounds on the other.
        """
        self._run_dinput_ground_truth(
            S=2, spatial=(16, 16), in_channels=4, out_channels=8,
            stride=(2, 1), padding="valid",
        )

    # ---- precision policy ------------------------------------

    @staticmethod
    def _capture(fn):
        """Runs fn and returns every UserWarning it emitted."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
        return [str(w.message) for w in caught]

    def _make_uncompiled_conv(self):
        """A Conv2d that has never seen _compile_for_device."""
        return Conv2d(
            in_channels=self.IN_CHANNELS,
            out_channels=self.OUT_CHANNELS,
            filter_size=self.FILTER_SIZE,
        )

    def _apply(self, layer, compute_dtype):
        return lambda: layer._apply_precision(config.DTypePolicy(compute_dtype))

    def test_policy_matching_the_bound_path_is_silent(self):
        """Asking for exactly what the bound kernels already do isn't a mismatch."""
        matching = self.layer._effective_compute_dtype()
        self.assertEqual(self._capture(self._apply(self.layer, matching)), [])

    def test_mismatched_policy_warns_once_naming_both_dtypes(self):
        """float64 is never an effective dtype, so it mismatches on either path."""
        effective = self.layer._effective_compute_dtype()
        caught = self._capture(self._apply(self.layer, "float64"))

        self.assertEqual(len(caught), 1)
        self.assertIn("float64", caught[0])
        self.assertIn(effective, caught[0])
        self.assertIn("Conv2d", caught[0])

    def test_repeated_compile_does_not_repeat_the_warning(self):
        """Model.to() then Model.finalize() both call _compile_for_device."""
        layer = self._make_uncompiled_conv()
        layer._apply_precision(config.DTypePolicy("float64"))

        caught = self._capture(
            lambda: [layer._compile_for_device(self.backend_name) for _ in range(2)]
        )
        self.assertEqual(len(caught), 1)

    def test_precision_set_before_compile_defers_to_compile(self):
        """set_precision() can precede to(); there's no binding to check yet."""
        layer = self._make_uncompiled_conv()

        before = self._capture(self._apply(layer, "float64"))
        self.assertEqual(before, [])

        after = self._capture(lambda: layer._compile_for_device(self.backend_name))
        self.assertEqual(len(after), 1)

    def test_ignored_policy_is_still_stored_for_serialization(self):
        """The warning must not stop the policy reaching Model.save()'s manifest."""
        self._capture(self._apply(self.layer, "float64"))

        self.assertEqual(self.layer.precision_policy.compute_dtype_name, "float64")

    def test_layer_still_computes_under_an_ignored_policy(self):
        """Warn-and-continue: output is unchanged, not an exception."""
        self._capture(self._apply(self.layer, "float64"))

        out = self.layer.forward(self.test_images, training=True)
        self.assertEqual(out.shape, (self.CONV_SHAPE[0], 28, 28, self.OUT_CHANNELS))
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(out))))

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


base_case.register_test_suites(globals(), TestConvLayer)