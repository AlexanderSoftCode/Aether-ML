"""
Aether-ML Unit Testing Template Skeleton
========================================
This template serves as a standardized blueprint for creating multi-backend
unit tests for layers, activations, and losses across NumPy (CPU) and CuPy (GPU).

Architecture Pattern:
- Dynamically creates test classes per backend target using factory metaprogramming (`make_suite`).
- Registers generated classes into `globals()` so `python3 -m unittest discover` picks them up.
- Enforces state isolation per backend using `set_backend()` and `AetherBaseTestCase`.
"""
import aether.config as config
import tests.base_case as base_case
# Replace this with your specific layout import
from aether.layers.conv import Conv2d

# Rewrite `TestLayer` for specific class being tested
# Also, rewrite the parent class based on if a layer is being tested
# or another component is being tested
class TestLayer(base_case.AetherBaseLayerTestCase):
    # Place any constants here
    INPUT_SHAPE = (28, 28, 1)
    NUM_FILTERS = 4
    FILTER_SIZE = (3, 3)
    STRIDE = (1, 1)
    PADDING = "same"
    def setUp(self):
        super().__init__()
        # These below are called in the parent class
        # self.backend_name = backend_name
        # Function from aether.config.py
        # config.set_backend(backend_name=self.backend_name)
        # self.xp = config.xp

        # Grab extra utility imports for specific backend if needed
        self.as_strided = config.get_stride_utility(self.xp)

        # Add the layerclass, then add its respective arguments after
        self.layer = self.make_built_layer(
            Conv2d, filter_size=self.FILTER_SIZE,
            stride=self.STRIDE, padding=self.PADDING,
            input_shape=self.INPUT_SHAPE
        )
        self.layer2 = self.make_built_layer(
            Conv2d, filter_size=self.FILTER_SIZE,
            stride=self.STRIDE, padding=self.PADDING,
            input_shape=self.INPUT_SHAPE
        )
        self.test_images = self.xp.random.randn(2, 28, 28, 1)

    def test_conv_forward_shape(self):
        """Verify output dimensions based on padding and stride"""
        output = self.layer.forward(self.test_images, training=True)
        # Given padding = 1, kernel = 3, 28x28 remainds 28x28 (same)

        expected_shape = (2, 28, 28, 4)
        self.assertEqual(output.shape, expected_shape)

    def test_conv_numerical_gradient_check(self):
        """Verify that analytical backpropagation calculations evaluate properly."""
        pass
    # Implement more tests if needed by creating more functions

# Below, you would add the helper function for the directiory to be discoverable, be sure
# to add the name of the test class, such that the make_suite function is setup
# base_case.register_test_suites(globals(), template_cls=TestLayer)
