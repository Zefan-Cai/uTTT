"""Import a Python module or package from a filesystem path.

torchtitan 0.1.0 (the pinned version) does not provide
``torchtitan.tools.utils.import_module_from_path``; this helper mirrors the
behaviour of the newer upstream function so that
``experimental.custom_model_path`` works with the pinned dependency.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from types import ModuleType


def import_module_from_path(path: str) -> ModuleType:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise ImportError(f"custom model path {path!r} does not exist")
    if os.path.isdir(path):
        init_file = os.path.join(path, "__init__.py")
        if not os.path.isfile(init_file):
            raise ImportError(f"custom model package {path!r} has no __init__.py")
        module_name = os.path.basename(os.path.normpath(path))
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[path]
        )
    else:
        module_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
