"""Stdlib-only bootstrap for scripts executed as ``python src/utils/<name>.py``."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def prepare_repo(script_file: str | Path) -> Path:
    p = Path(script_file).resolve()
    root = p.parent.parent.parent
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(root)
    return root
