import tests.base_case as base_case
from aether.optimizers.adam import Adam
from tests.optimizers.adam_base_suite import BaseTestOptimizerAdam


class TestAdam(BaseTestOptimizerAdam):
    OPTIMIZER_CLASS = Adam


base_case.register_test_suites(globals(), TestAdam)