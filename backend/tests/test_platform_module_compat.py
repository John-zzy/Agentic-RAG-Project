from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_backend_platform_package_remains_compatible_with_stdlib_platform_imports() -> None:
    backend_dir = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-c",
        (
            "import platform; "
            "print(hasattr(platform, 'python_implementation')); "
            "print(platform.python_implementation())"
        ),
    ]

    result = subprocess.run(
        command,
        cwd=backend_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["True", "CPython"]
