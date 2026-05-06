"""Run 01_annotation_gpt_annotation.py with --source public."""

from __future__ import annotations

import os
import subprocess
import sys

from utils.workspace import repository_root

ROOT = repository_root()
ANNOTATOR = ROOT / "src" / "01_annotation_gpt_annotation.py"


def _argv_without_source(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--source":
            i += 2
            continue
        if a.startswith("--source="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def main() -> None:
    if not ANNOTATOR.is_file():
        print(f"not found: {ANNOTATOR}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["GPT_ANNOTATION_SOURCE"] = "public"

    rest = _argv_without_source(list(sys.argv[1:]))
    cmd = [
        sys.executable,
        str(ANNOTATOR),
        "--source",
        "public",
        *rest,
    ]
    raise SystemExit(subprocess.call(cmd, env=env, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
