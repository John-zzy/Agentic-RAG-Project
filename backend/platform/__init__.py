"""Backend platform package with stdlib platform compatibility."""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path


def _load_stdlib_platform_module() -> object:
    stdlib_dir = Path(sysconfig.get_path("stdlib"))
    platform_py = stdlib_dir / "platform.py"
    spec = importlib.util.spec_from_file_location("_stdlib_platform", platform_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load stdlib platform module from {platform_py}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_platform = _load_stdlib_platform_module()

for _name in dir(_stdlib_platform):
    if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
        continue
    globals()[_name] = getattr(_stdlib_platform, _name)

__all__ = getattr(_stdlib_platform, "__all__", [])
