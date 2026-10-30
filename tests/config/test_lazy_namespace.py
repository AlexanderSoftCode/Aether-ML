import ast
import importlib
import pathlib

import aether
import tests.base_case as base_case

AETHER_ROOT = pathlib.Path(aether.__file__).parent


def _stub_exports(stub_path):
    """(re-exported names, __all__) declared by a .pyi stub."""
    tree = ast.parse(stub_path.read_text())
    reexported = {
        alias.asname
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names if alias.asname == alias.name
    }
    declared_all = next(
        set(ast.literal_eval(node.value))
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "__all__" for target in node.targets)
    )
    return reexported, declared_all


class TestLazyNamespace(base_case.AetherBaseTestCase):
    """Every package with an __init__.pyi: the lazy map and the stub list the same
    public names, and each one resolves."""

    def test_lazy_map_matches_stub_and_resolves(self):
        stubs = sorted(AETHER_ROOT.rglob("__init__.pyi"))
        self.assertTrue(stubs, "no __init__.pyi stubs found")

        for stub in stubs:
            package = ".".join(("aether", *stub.parent.relative_to(AETHER_ROOT).parts))
            with self.subTest(package=package):
                module = importlib.import_module(package)
                runtime_all = set(module.__all__)
                reexported, declared_all = _stub_exports(stub)

                self.assertEqual(reexported, declared_all,
                                 f"{stub.name}: imports and __all__ disagree")
                self.assertEqual(runtime_all, declared_all,
                                 "lazy map and stub list different public names")
                unresolved = [name for name in sorted(runtime_all)
                              if not hasattr(module, name)]
                self.assertEqual(unresolved, [], "lazy names that fail to resolve")


base_case.register_test_suites(globals(), template_cls=TestLazyNamespace)
