import importlib
  
def install_lazy_attrs(module_globals, name_map):
    """
    Wire up a lazy __getattr__/__dir__ pair into the CALLING package's
    module namespace.
 
    Parameters
    ----------
    module_globals : dict
        Must be the literal `globals()` of the __init__.py calling this
        function. install_lazy_attrs writes __getattr__/__dir__/__all__
        directly into that dict, which is what actually makes them live
        on that specific package -- __getattr__ is resolved per-module,
        not inherited up the package hierarchy, so every __init__.py that
        wants lazy attribute resolution needs to call this itself.
    name_map : dict[str, str]
        {public_name: "fully.qualified.module.path"}. On first access of
        `package.public_name`, the target module is imported and the
        attribute is pulled off of it.
 
    Notes
    -----
    The resolved value is cached directly into module_globals on first
    access (module_globals[name] = value), so __getattr__ is only ever
    invoked once per name -- every subsequent access is a plain,
    zero-overhead attribute lookup, same as if it had been imported
    eagerly. Also, While Python 3.15 PEP 810 - Explicit Lazy Imports 
    circumvents the need for a file like this, its no need for users
    to be locked out of this package for not being on python >=3.15
    """
    module_name = module_globals.get("__name__", "<unknown>")
 
    # Merge rather than clobber: a package may want a handful of names
    # eagerly imported above the install_lazy_attrs() call (e.g. a
    # version string) in addition to the lazy ones.
    existing_all = set(module_globals.get("__all__", ()))
    module_globals["__all__"] = sorted(existing_all | set(name_map))
 
    def __getattr__(name):
        if name in name_map:
            target_module = importlib.import_module(name_map[name])
            value = getattr(target_module, name)
            module_globals[name] = value  # cache: next access is a plain lookup
            return value
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
 
    def __dir__():
        return sorted(set(module_globals.keys()) | set(name_map))
 
    module_globals["__getattr__"] = __getattr__
    module_globals["__dir__"] = __dir__