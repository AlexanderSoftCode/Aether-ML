
import numpy as np

import tests.base_case as base_case
from aether.layers.conv import Conv2d
from tests.layers.conv2d.base import ConvLayerTestBase


class TestConvBackward(ConvLayerTestBase):
    """Backward pass: gradient shapes, finite differences, dX ground truth."""

    def test_backward_gradient_shapes(self):
        """Verify backprop gradients match weight shapes"""
        output = self.layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(*output.shape) #mock gradient
        self.layer.backward(dvalues)

        self.assertEqual(self.layer.dweights.shape, self.layer.filter_weights.shape)
        self.assertEqual(self.layer.dbiases.shape, self.layer.biases.shape)

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
        layer.forward(fixed_input, training=True)
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

    def test_same_padding_even_filter_numerical_gradient(self):
        """Fallback dweights match finite differences on an even, strided filter."""
        layer = self._make_same_layer((7, 7), (2, 2), (2, 2), in_channels=1, out_channels=2)
        x = self.xp.random.randn(2, 7, 7, 1)
        layer.forward(x, training=True)
        layer.backward(self.xp.ones_like(layer.output))
        analytical = self._to_numpy(layer.dweights)

        eps = 1e-2
        numerical = np.zeros_like(analytical)
        for idx in np.ndindex(*analytical.shape):
            layer.filter_weights[idx] += eps
            layer.invalidate_shadow_caches()
            plus = float(self.xp.sum(layer.forward(x, training=False)))
            layer.filter_weights[idx] -= 2 * eps
            layer.invalidate_shadow_caches()
            minus = float(self.xp.sum(layer.forward(x, training=False)))
            layer.filter_weights[idx] += eps
            layer.invalidate_shadow_caches()
            numerical[idx] = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(analytical, numerical, rtol=3e-2, atol=2e-2)

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
        fH, fW, C_in, _ = weights.shape
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

        layer._forward_fallback(inputs, training=True)
        layer._backward_fallback(dvalues)
        self._assert_gemm_close(layer.dinputs, truth, "dinputs (fallback vs truth)",
                                tol=1e-5)

        from aether.custom_kernels import conv_kernel
        if self.backend_name == 'cupy' and conv_kernel.get_is_conv_gpu_available():
            layer._forward_gpu(inputs, training=True)
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

    def test_conv_dinput_ground_truth_even_filter(self):
        """Even filter, stride 1: the extra pad pixel sits on the bottom/right."""
        self._run_dinput_ground_truth(
            S=2, spatial=(8, 8), in_channels=4, out_channels=8,
            stride=(1, 1), padding="same", filter_size=(2, 2),
        )

    def test_conv_dinput_ground_truth_even_filter_strided(self):
        """Even filter, stride 2 on an odd input."""
        self._run_dinput_ground_truth(
            S=2, spatial=(9, 9), in_channels=4, out_channels=8,
            stride=(2, 2), padding="same", filter_size=(4, 4),
        )

    def test_backward_without_input_grad_keeps_parameter_grads(self):
        """needs_input_grad=False drops dX only; dW/dB must match a normal backward."""
        output = self.layer.forward(self.test_images, training=True)
        dvalues = self.xp.random.randn(*output.shape).astype(self.xp.float32)

        self.assertIsNotNone(self.layer.backward(dvalues))
        expected_dw, expected_db = self.layer.dweights.copy(), self.layer.dbiases.copy()

        self.layer.needs_input_grad = False
        self.assertIsNone(self.layer.backward(dvalues))
        self.assertIsNone(self.layer.dinputs)
        # GPU dW/dB accumulate split-K slices with atomics, so summation order
        # varies run to run. Compare against the tensor's scale, not per
        # element: an entry that cancels to near zero still carries the
        # rounding of its O(10) partial sums, which a plain rtol rejects.
        self._assert_gemm_close(self.layer.dweights, expected_dw, "dweights", tol=1e-5)
        self._assert_gemm_close(self.layer.dbiases, expected_db, "dbiases", tol=1e-5)


base_case.register_test_suites(globals(), TestConvBackward)
