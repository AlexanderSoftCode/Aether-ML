import warnings

import numpy as np

import aether.config as config
import tests.base_case as base_case

from aether.layers import Dense, ReLU
from aether.losses import SoftmaxCategoricalCrossEntropy
from aether.metrics import CategoricalAccuracy
from aether.model import Model
from aether.optimizers import Adam
from aether.preprocessing.transforms import (
    Compose,
    Preprocess,
    Rescale,
    StandardScaler,
    ToTensor,
)
N_SAMPLES = 2048
N_FEATURES = 512
NUM_CLASSES = 2


def _float16_policy():
    """Builds a float16 policy without the CPU-emulation warning leaking into output."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*NumPy float16 is emulated.*", category=UserWarning
        )
        return config.DTypePolicy(compute_dtype="float16")


class TestPrecisionExemptionBase(base_case.AetherBaseTestCase):
    __test__ = False

    def setUp(self):
        super().setUp()
        np.random.seed(7)
        self.raw_X = np.random.randint(
            0, 256, size=(N_SAMPLES, N_FEATURES), dtype=np.uint8
        )

    def set_precision(self, model, compute_dtype="float16"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*NumPy float16 is emulated.*", category=UserWarning
            )
            model.set_precision(compute_dtype)

    # ---- Exemption flags ----

    def test_preprocess_base_is_not_precision_exempt(self):
        self.assertFalse(Preprocess._precision_exempt)
        self.assertFalse(Compose([])._precision_exempt)
        self.assertFalse(ToTensor()._precision_exempt)
        self.assertFalse(Rescale()._precision_exempt)

    def test_standard_scaler_is_precision_exempt(self):
        self.assertTrue(StandardScaler()._precision_exempt)

    def test_compose_skips_exempt_member_but_still_reaches_others(self):
        """Compose mirrors Model.set_precision's layer loop, member by member."""
        to_tensor = ToTensor()
        scaler = StandardScaler()
        Compose([to_tensor, scaler])._apply_precision(_float16_policy())

        self.assertEqual(to_tensor.dtype, "float16")
        self.assertIsNone(scaler.dtype)

    def test_nested_compose_still_recurses(self):
        inner_to_tensor = ToTensor()
        inner_scaler = StandardScaler()
        outer = Compose([Compose([inner_to_tensor, inner_scaler])])
        outer._apply_precision(_float16_policy())

        self.assertEqual(inner_to_tensor.dtype, "float16")
        self.assertIsNone(inner_scaler.dtype)

    # ---- The regression itself ----

    def test_float16_policy_leaves_compose_scaler_statistics_finite(self):
        """The bug: a float16 policy drove cp/np.std to inf, zeroing every batch."""
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            StandardScaler(),
        ])
        pipeline._apply_precision(_float16_policy())
        pipeline.fit(self.raw_X)

        scaler = pipeline.transforms[-1]
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(scaler.mean))))
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(scaler.std))))
        self.assertGreater(float(scaler.std.ravel()[0]), 0.0)

    def test_float16_policy_pipeline_output_is_not_degenerate(self):
        pipeline = Compose([
            ToTensor(dtype="float32", target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            StandardScaler(),
        ])
        pipeline._apply_precision(_float16_policy())
        out = pipeline.fit(self.raw_X)(self.raw_X)

        self.assertTrue(bool(self.xp.all(self.xp.isfinite(out))))
        # Division by an inf std collapsed every element to exactly 0.0.
        self.assertGreater(float(self.xp.var(out.astype(self.xp.float32))), 0.5)

    def test_float16_policy_with_unpinned_to_tensor_stays_finite(self):
        """Belt-and-braces: the policy reaches the scaler's *input* via ToTensor,
        so the stats are reduced over a float16 array even with the exemption."""
        pipeline = Compose([
            ToTensor(target_device=self.backend_name),
            Rescale(factor=1.0 / 255.0),
            StandardScaler(),
        ])
        pipeline._apply_precision(_float16_policy())
        out = pipeline.fit(self.raw_X)(self.raw_X)

        scaler = pipeline.transforms[-1]
        self.assertTrue(bool(self.xp.all(self.xp.isfinite(scaler.std))))
        self.assertGreater(float(self.xp.var(out.astype(self.xp.float32))), 0.5)

    def test_float16_dtype_pin_is_rejected_outright(self):
        """StandardScaler never allows a sub-float32 dtype pin, exemption or not --
        mean/std are never computed or stored below single precision, so a float16
        pin can't even be constructed regardless of set_precision()."""
        with self.assertRaises(ValueError):
            StandardScaler(dtype="float16")

    # ---- Model-level dispatch ----

    def test_model_set_precision_skips_bare_standard_scaler(self):
        model = Model()
        model.add(Dense(N_FEATURES, NUM_CLASSES))
        scaler = StandardScaler()
        model.configure(preprocessor=scaler)
        self.set_precision(model)

        self.assertIsNone(scaler.dtype)

    def test_finalize_redispatch_skips_bare_standard_scaler(self):
        """set_precision() before configure() re-dispatches from finalize()."""
        model = Model()
        model.add(Dense(N_FEATURES, NUM_CLASSES))
        self.set_precision(model)

        scaler = StandardScaler()
        model.configure(preprocessor=scaler)
        model.to(self.backend_name)
        model.finalize((N_FEATURES,))

        self.assertIsNone(scaler.dtype)

    def test_float16_model_with_pipeline_does_not_collapse_to_one_class(self):
        """End-to-end shape of the reported bug: every image predicted as one class."""
        y = (np.arange(N_SAMPLES) % NUM_CLASSES).astype(np.int32)
        raw_X = self.raw_X.astype(np.int16)
        raw_X[y == 1] = np.clip(raw_X[y == 1] + 90, 0, 255)
        raw_X = raw_X.astype(np.uint8)

        model = Model()
        model.manual_seed(0)
        model.add(Dense(N_FEATURES, 16))
        model.add(ReLU())
        model.add(Dense(16, NUM_CLASSES))
        model.configure(
            loss=SoftmaxCategoricalCrossEntropy(),
            optimizer=Adam(lr=0.01),
            accuracy=CategoricalAccuracy(),
            preprocessor=Compose([
                ToTensor(dtype="float32", target_device=self.backend_name),
                Rescale(factor=1.0 / 255.0),
                StandardScaler(),
            ]),
        )
        model.to(self.backend_name)
        self.set_precision(model)
        model.finalize((N_FEATURES,))
        model.train(
            raw_X, y, epochs=3, batch_size=256, verbose=0, print_every=0
        )

        preds = model.predict(raw_X, batch_size=256)
        self.assertGreater(len(np.unique(np.argmax(preds, axis=1))), 1)

        _, acc = model.evaluate(raw_X, y, batch_size=256, verbose=0)
        self.assertGreater(acc, 0.6)


base_case.register_test_suites(globals(), TestPrecisionExemptionBase)
