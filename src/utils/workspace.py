"""Repository layout, canonical artifact paths, and runtime environment for batch scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "repository_root",
    "repository_root_for_script",
    "prepare_analysis_environment",
    "gpt_public_annotation_detailed_csv",
    "public_list_pronouns_detailed_csv",
    "filtering_processed_dir",
]


def repository_root_for_script(script_file: str | Path) -> Path:
    """Resolve repo root from ``<repo>/src/foo.py`` or ``<repo>/src/utils/foo.py``."""
    p = Path(script_file).resolve()
    if p.parent.name == "utils":
        return p.parent.parent.parent
    return p.parent.parent


def repository_root() -> Path:
    """Repo root inferred from this module's path (``<repo>/src/utils/workspace.py``)."""
    return Path(__file__).resolve().parent.parent.parent


def prepare_analysis_environment(
    script_file: str | Path,
    *,
    matplotlib_backend: str | None = "Agg",
) -> Path:
    """``chdir`` to repo root, prepend ``<root>/src`` to ``sys.path``, optionally set a matp..."""
    root = repository_root_for_script(script_file)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(root)
    if matplotlib_backend:
        import matplotlib

        matplotlib.use(matplotlib_backend)
    return root


def gpt_public_annotation_detailed_csv(root: Path | None = None) -> Path:
    """Default GPT sentence-level annotation table for public-list runs."""
    r = root if root is not None else repository_root()
    return r / "data" / "processed" / "gpt_annotation_public_run" / "gpt_annotation_detailed.csv"


def public_list_pronouns_detailed_csv(root: Path | None = None) -> Path:
    """Pronoun detection export restricted to the public-list poem IDs."""
    r = root if root is not None else repository_root()
    return r / "data" / "processed" / "ukrainian_pronouns_detailed_public_list.csv"


def filtering_processed_dir(root: Path | None = None) -> Path:
    """Output directory for stanza / sub-poem preprocessing (layer 0–1)."""
    r = root if root is not None else repository_root()
    return r / "data" / "To_run" / "00_filtering"
