import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.layers.conv import Conv2d


class ConvLayerTestBase(base_case.AetherBaseLayerTestCase):
    """Shared fixture and comparison helpers for the Conv2d test files."""
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

    SAME_CASES = (
        # (spatial, filter_size, stride)
        ((8, 8), (2, 2), (1, 1)),
        ((8, 8), (4, 4), (2, 2)),
        ((9, 7), (2, 4), (2, 1)),
        ((8, 8), (3, 3), (2, 2)),
        ((7, 7), (3, 3), (2, 2)),
        ((8, 8), (1, 1), (3, 3)),
    )

    def _make_same_layer(self, spatial, filter_size, stride, in_channels=2, out_channels=3):
        return self.make_built_layer(
            Conv2d, input_shape=(*spatial, in_channels), seed=self.SEED,
            in_channels=in_channels, out_channels=out_channels,
            filter_size=filter_size, stride=stride, padding="same",
        )
