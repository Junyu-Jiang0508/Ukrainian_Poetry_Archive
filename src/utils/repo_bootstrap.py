"""Stdlib-only bootstrap for scripts executed as ``python src/.../<name>.py``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT_MARKERS = ("requirements.txt", ".git")


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    raise RuntimeError(f"could not locate repo root from {start}")


def prepare_repo(script_file: str | Path) -> Path:
    p = Path(script_file).resolve()
    root = _find_repo_root(p.parent)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(root)
    return root
