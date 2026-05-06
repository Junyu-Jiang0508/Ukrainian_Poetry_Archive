"""Stdlib-only bootstrap for scripts executed as ``python src/utils/<name>.py``.

In that execution mode ``sys.path[0]`` is ``.../src/utils``, so the ``utils``
package is not importable until ``<repo>/src`` is prepended. Import this module
first (same directory), call ``prepare_repo``, then ``from utils.…``.
"""

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
