"""
Simple null objects that are used by the Model class.
Avoids making us write needless branch conditionals.
"""
from aether.preprocessing.transforms import Preprocess


class NullAccuracy:
    def calculate(self, *args, **kwargs):
        return 0.0

    def calculate_accumulated(self):
        return 0.0

    def new_pass(self):
        pass

    def init(self, y):
        pass


class NullOptimizer:
    def step(self):
        pass

class _NullProgress:
    """
    No-op stand-in for ``TrainingProgress`` used when ``verbose == 0``.
    """

    __slots__ = ()

    def start_epoch(self, epoch: int) -> None:
        return

    def tick(self, step: int, force: bool = False) -> None:
        return

    def update_metrics(
        self,
        step: int,
        loss: float,
        acc: float,
        lr: float,
        reg_loss: float | None = None,
    ) -> None:
        return

    def commit_epoch(
        self,
        epoch: int,
        epoch_loss: float,
        epoch_acc: float,
        lr: float,
    ) -> None:
        return

    def commit_validation(self, val_loss: float, val_acc: float) -> None:
        return

    def close(self) -> None:
        return

class NullPreprocessor(Preprocess):
    """Null-object preprocessor for models with no attached preprocessing pipeline.

    A pure identity: ``transform()`` hands back exactly what it was given, and the
    device/precision hooks are genuine no-ops. Model.finalize() installs this
    unconditionally so the per-batch dispatch can call ``.transform()`` with zero
    ``is not None`` branching in the hot loop -- the same role NullOptimizer and
    NullAccuracy play elsewhere in the codebase.

    Deliberately does NOT absorb device migration or precision casting. Every
    finalized model owns one of these, so a migrating null object would silently
    add a host->device copy to every batch of every pipeline-free model -- exactly
    the per-step PCIe traffic that Model._assert_device_alignment exists to forbid.
    Device strictness stays with that guard; device migration stays with an
    explicit ToTensor the user opted into.

    Trivially fit, so it never triggers the implicit-fit path in train().
    """
    is_fitted = True

    def _compile_for_device(self, device):
        return

    def _apply_precision(self, policy):
        return

    def transform(self, X):
        return X
