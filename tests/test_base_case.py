import unittest
import warnings

import numpy as np

import aether.config as config
import tests.base_case as base_case
from aether.base import Layer


class DummyLayer(Layer):
    """Minimal concrete Layer for exercising make_built_layer/set_precision
    plumbing without pulling in a real layer's forward/backward math."""

    def __init__(self, multiplier=1.0):
        super().__init__()
        self.multiplier = multiplier
        self.compiled_device = None
        self.applied_policy = None

    def build(self, input_shape, seed=None):
        super().build(input_shape)
        if seed is not None:
            self.seed = seed
        xp = config.xp
        self.weights = xp.zeros(input_shape, dtype=xp.float32)
        return self.output_shape

    def _compile_for_device(self, device):
        self.compiled_device = device

    def _apply_precision(self, policy):
        self.applied_policy = policy

    def forward(self, inputs, training):
        return inputs * self.multiplier

    def backward(self, dvalues):
        return dvalues * self.multiplier


class DummyExemptLayer(DummyLayer):
    """Mirrors layers like SoftMax/BatchNorm that opt out of precision casting."""
    _precision_exempt = True


class MinimalLayer(Layer):
    """A Layer that relies entirely on the base class's default (no-op)
    _compile_for_device hook -- verifies make_built_layer tolerates layers
    that don't override it."""

    def build(self, input_shape, seed=None):
        super().build(input_shape)
        return self.output_shape

    def forward(self, inputs, training):
        return inputs

    def backward(self, dvalues):
        return dvalues


class OrderTrackingLayer(Layer):
    """Records call order so make_built_layer's documented contract
    (compile-for-device happens before build) can be verified directly,
    rather than inferred indirectly."""

    def __init__(self, call_log):
        super().__init__()
        self.call_log = call_log

    def _compile_for_device(self, device):
        self.call_log.append(("compile", device))

    def build(self, input_shape, seed=None):
        super().build(input_shape)
        self.call_log.append(("build", input_shape))
        return self.output_shape

    def forward(self, inputs, training):
        return inputs

    def backward(self, dvalues):
        return dvalues


class DummyComponent:
    """Minimal non-layer component (mirrors the shape of Loss/Optimizer/
    Accuracy) for exercising make_component."""

    def __init__(self, value=1):
        self.value = value
        self.compiled_device = None

    def _compile_for_device(self, device):
        self.compiled_device = device


class DummyComponentNoDeviceHook:
    """Deliberately missing _compile_for_device, to verify make_component's
    hasattr() guard doesn't assume every component has the hook."""

    def __init__(self, value=1):
        self.value = value



def _make_probe_case(backend_name, layer=False):
    """Returns a ready-to-use (setUp() already called) probe test-case
    instance bound to the given backend_name. Caller must call
    instance.tearDown() when done (tests below do this in `finally`)."""
    parent = base_case.AetherBaseLayerTestCase if layer else base_case.AetherBaseTestCase
    probe_cls = type("_ProbeCase", (parent,), {"test_probe": lambda self: None})
    instance = probe_cls("test_probe")
    instance.backend_name = backend_name
    instance.setUp()
    return instance


# ---- AetherBaseTestCase: backend routing (setUp/tearDown) -----------

class TestAetherBaseTestCaseBackendRouting(unittest.TestCase):
    """Direct replacement for the old test_backend_pointer_swap: verifies
    setUp() binds config.xp/self.xp to the backend named by self.backend_name,
    and tearDown() always resets the global backend back to numpy."""

    def test_default_backend_name_is_numpy(self):
        self.assertEqual(base_case.AetherBaseTestCase.backend_name, "numpy")

    def test_setup_binds_numpy_backend(self):
        case = _make_probe_case("numpy")
        try:
            self.assertIs(case.xp, np)
            self.assertEqual(config.xp.__name__, "numpy")
        finally:
            case.tearDown()

    def test_setup_binds_cupy_backend(self):
        if not config.HAS_CUPY:
            self.skipTest("CuPy not available in this environment.")
        import cupy as cp

        case = _make_probe_case("cupy")
        try:
            self.assertIs(case.xp, cp)
            self.assertEqual(config.xp.__name__, "cupy")
        finally:
            case.tearDown()

    def test_teardown_resets_to_numpy_after_cupy(self):
        if not config.HAS_CUPY:
            self.skipTest("CuPy not available in this environment.")

        case = _make_probe_case("cupy")
        case.tearDown()
        self.assertEqual(config.xp.__name__, "numpy")

    def test_shortDescription_suppressed_for_verbose_output(self):
        case = _make_probe_case("numpy")
        try:
            self.assertIsNone(case.shortDescription())
        finally:
            case.tearDown()


class TestEndToEndBackendNamingInvariant(unittest.TestCase):
    """Closes the loop that the old test_backend_pointer_swap actually
    checked: that a class generated by register_test_suites resolves
    self.xp to the backend implied by its generated `_NUMPY`/`_CUPY` name
    suffix. Run once here instead of once per consuming test file."""

    def test_generated_subclass_xp_matches_name_suffix(self):
        class Template(base_case.AetherBaseTestCase):
            def test_probe(self):
                pass

        fake_globals = {}
        base_case.register_test_suites(fake_globals, Template)

        for backend in base_case.BACKENDS_TO_TEST:
            cls = fake_globals[f"Template_{backend.upper()}"]
            instance = cls("test_probe")
            instance.setUp()
            try:
                if "CUPY" in cls.__name__.upper():
                    self.assertEqual(instance.xp.__name__, "cupy")
                else:
                    self.assertEqual(instance.xp.__name__, "numpy")
            finally:
                instance.tearDown()

# ---- AetherBaseLayerTestCase: make_built_layer / make_component / set_precision ---------

class TestMakeBuiltLayer(unittest.TestCase):
    """Exercises AetherBaseLayerTestCase.make_built_layer's
    device-compile -> build lifecycle."""

    def test_triggers_device_compile_before_build(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            call_log = []
            case.make_built_layer(OrderTrackingLayer, input_shape=(4,), call_log=call_log)
            self.assertEqual(call_log, [("compile", "numpy"), ("build", (4,))])
        finally:
            case.tearDown()

    def test_allocates_expected_weight_shape(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            layer = case.make_built_layer(DummyLayer, input_shape=(4,))
            self.assertEqual(layer.compiled_device, "numpy")
            self.assertIsNotNone(layer.weights)
            self.assertEqual(layer.weights.shape, (4,))
        finally:
            case.tearDown()

    def test_forwards_seed_and_kwargs_to_layer(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            layer = case.make_built_layer(DummyLayer, input_shape=(2,), seed=123, multiplier=3.0)
            self.assertEqual(layer.seed, 123)
            self.assertEqual(layer.multiplier, 3.0)
        finally:
            case.tearDown()

    def test_seed_defaults_to_none_when_omitted(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            layer = case.make_built_layer(DummyLayer, input_shape=(2,))
            self.assertIsNone(layer.seed)
        finally:
            case.tearDown()

    def test_tolerates_layer_without_device_hook_override(self):
        """MinimalLayer relies on Layer's default no-op _compile_for_device;
        make_built_layer must not choke since hasattr() is always True for
        any Layer subclass (the hook exists on the base class itself)."""
        case = _make_probe_case("numpy", layer=True)
        try:
            layer = case.make_built_layer(MinimalLayer, input_shape=(5,))
            self.assertEqual(layer.output_shape, (5,))
        finally:
            case.tearDown()


class TestMakeComponent(unittest.TestCase):
    """Exercises AetherBaseLayerTestCase.make_component (used for
    Loss/Optimizer/Accuracy-style non-layer components)."""

    def test_triggers_compile_for_device(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            comp = case.make_component(DummyComponent, value=5)
            self.assertEqual(comp.compiled_device, "numpy")
            self.assertEqual(comp.value, 5)
        finally:
            case.tearDown()

    def test_tolerates_component_missing_device_hook(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            comp = case.make_component(DummyComponentNoDeviceHook, value=7)
            self.assertEqual(comp.value, 7)
            self.assertFalse(hasattr(comp, "compiled_device"))
        finally:
            case.tearDown()


class TestSetPrecision(unittest.TestCase):
    """Exercises AetherBaseLayerTestCase.set_precision, including the
    _precision_exempt bypass and the float16-emulation warning suppression."""

    def test_applies_policy_to_non_exempt_layer(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            layer = DummyLayer()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                policy = case.set_precision(layer, compute_dtype="float16")
            self.assertIs(layer.applied_policy, policy)
            self.assertEqual(policy.compute_dtype_name, "float16")
        finally:
            case.tearDown()

    def test_skips_exempt_layer(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            layer = DummyExemptLayer()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                case.set_precision(layer, compute_dtype="float16")
            self.assertIsNone(layer.applied_policy)
        finally:
            case.tearDown()

    def test_suppresses_float16_emulation_warning(self):
        """On the NumPy backend, DTypePolicy('float16') normally emits a
        UserWarning about emulation -- set_precision must swallow it."""
        case = _make_probe_case("numpy", layer=True)
        try:
            layer = DummyLayer()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                case.set_precision(layer, compute_dtype="float16")
            emulation_warnings = [w for w in caught if "emulated" in str(w.message)]
            self.assertEqual(emulation_warnings, [])
        finally:
            case.tearDown()

    def test_returns_policy_for_direct_use(self):
        case = _make_probe_case("numpy", layer=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                policy = case.set_precision(DummyLayer(), compute_dtype="float32")
            self.assertEqual(policy.compute_dtype_name, "float32")
        finally:
            case.tearDown()


# ---- __init_subclass__ discovery-flag assignment -----------------

class TestDiscoveryFlagAssignment(unittest.TestCase):
    """Verifies the __test__ flag convention: any class with "Base" in its
    name is marked non-discoverable, everything else defaults to
    discoverable. (This alone doesn't stop stdlib unittest from collecting
    a leaked base class -- see register_test_suites' MRO purge below for
    the actual enforcement.)"""

    def test_base_classes_flagged_non_discoverable(self):
        self.assertFalse(base_case.AetherBaseTestCase.__test__)
        self.assertFalse(base_case.AetherBaseLayerTestCase.__test__)

    def test_concrete_subclass_without_base_in_name_is_discoverable(self):
        cls = type("ProbeConcreteCase", (base_case.AetherBaseTestCase,), {})
        self.assertTrue(cls.__test__)

    def test_subclass_with_base_in_name_remains_non_discoverable(self):
        cls = type("SomeBaseCase", (base_case.AetherBaseTestCase,), {})
        self.assertFalse(cls.__test__)

    def test_explicit_test_false_is_overridden_when_name_lacks_base(self):
        """__init_subclass__ runs after the class body executes, so it wins
        over any explicit `__test__ = False` set in the body unless the
        name itself contains "Base" -- this is intentional (name is the
        single source of truth), but worth pinning down explicitly since
        it's easy to assume the body's literal value always wins."""
        cls = type("ProbeExplicitFalse", (base_case.AetherBaseTestCase,), {"__test__": False})
        self.assertTrue(cls.__test__)


# ---- register_test_suites ---------------------

class TestRegisterTestSuites(unittest.TestCase):
    """Verifies backend fan-out generation and the ancestor-leak purge
    logic that protects consuming test files from the
    discoverable-base-class bug described in base_case.py's docstring."""

    def test_generates_one_subclass_per_backend(self):
        class Template(base_case.AetherBaseTestCase):
            def test_dummy(self):
                pass

        fake_globals = {"Template": Template}
        base_case.register_test_suites(fake_globals, Template)

        expected_names = {f"Template_{b.upper()}" for b in base_case.BACKENDS_TO_TEST}
        self.assertEqual(set(fake_globals.keys()), expected_names)

    def test_generated_subclasses_have_correct_backend_name_and_test_flag(self):
        class Template(base_case.AetherBaseTestCase):
            def test_dummy(self):
                pass

        fake_globals = {}
        base_case.register_test_suites(fake_globals, Template)

        for backend in base_case.BACKENDS_TO_TEST:
            cls = fake_globals[f"Template_{backend.upper()}"]
            self.assertEqual(cls.backend_name, backend)
            self.assertTrue(cls.__test__)
            self.assertTrue(issubclass(cls, Template))

    def test_purges_leaked_ancestor_base_classes_from_globals(self):
        """Simulates the exact leak this whole mechanism exists to prevent:
        a test module doing `from tests.base_case import
        AetherBaseLayerTestCase` at module scope, binding the ancestor as a
        bare, independently-discoverable TestCase subclass."""

        class Template(base_case.AetherBaseLayerTestCase):
            def test_dummy(self):
                pass

        fake_globals = {
            "AetherBaseLayerTestCase": base_case.AetherBaseLayerTestCase,
            "AetherBaseTestCase": base_case.AetherBaseTestCase,
            "Template": Template,
        }
        base_case.register_test_suites(fake_globals, Template)

        self.assertNotIn("AetherBaseLayerTestCase", fake_globals)
        self.assertNotIn("AetherBaseTestCase", fake_globals)
        self.assertNotIn("Template", fake_globals)

    def test_does_not_purge_unrelated_same_named_object(self):
        """Identity check: a global that merely shares an ancestor's class
        *name* but is a different object must be left alone."""

        class ImposterAetherBaseTestCase:
            pass

        class Template(base_case.AetherBaseTestCase):
            def test_dummy(self):
                pass

        fake_globals = {
            "AetherBaseTestCase": ImposterAetherBaseTestCase,
            "Template": Template,
        }
        base_case.register_test_suites(fake_globals, Template)

        self.assertIs(fake_globals["AetherBaseTestCase"], ImposterAetherBaseTestCase)

    def test_purge_ignores_ancestors_without_explicit_test_false_in_own_dict(self):
        """unittest.TestCase itself is an ancestor of every template but
        never sets __test__ = False in its own __dict__, so it must never
        be treated as a leak candidate."""

        class Template(base_case.AetherBaseTestCase):
            def test_dummy(self):
                pass

        fake_globals = {"TestCase": unittest.TestCase, "Template": Template}
        base_case.register_test_suites(fake_globals, Template)

        self.assertIn("TestCase", fake_globals)
        self.assertIs(fake_globals["TestCase"], unittest.TestCase)

    def test_safe_to_call_when_ancestor_names_absent_from_globals(self):
        """Normal/expected case (qualified `import tests.base_case as
        base_case` style): no ancestor names present at all -- must not
        raise KeyError or similar."""

        class Template(base_case.AetherBaseLayerTestCase):
            def test_dummy(self):
                pass

        fake_globals = {}
        try:
            base_case.register_test_suites(fake_globals, Template)
        except KeyError:
            self.fail("register_test_suites raised KeyError on an empty globals dict.")

        expected_names = {f"Template_{b.upper()}" for b in base_case.BACKENDS_TO_TEST}
        self.assertEqual(set(fake_globals.keys()), expected_names)