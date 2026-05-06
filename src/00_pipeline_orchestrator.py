"""Run numbered pipeline stages with one consistent CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils.pipeline_catalog import build_pipeline_catalog
from utils.workspace import repository_root_for_script

ROOT = repository_root_for_script(__file__)


def _select_stage_ids(args: argparse.Namespace, ordered_ids: list[str]) -> set[str]:
    if args.only:
        return {x.strip().lower() for x in args.only}

    start = args.from_stage.lower() if args.from_stage else ordered_ids[0]
    end = args.to_stage.lower() if args.to_stage else ordered_ids[-1]
    if start not in ordered_ids:
        raise ValueError(f"--from-stage not found: {args.from_stage}")
    if end not in ordered_ids:
        raise ValueError(f"--to-stage not found: {args.to_stage}")
    i0 = ordered_ids.index(start)
    i1 = ordered_ids.index(end)
    if i0 > i1:
        raise ValueError("--from-stage must be before --to-stage in pipeline order")
    return set(ordered_ids[i0 : i1 + 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified runner for stages 00a..18.",
    )
    parser.add_argument("--list", action="store_true", help="List stages and exit.")
    parser.add_argument("--from-stage", dest="from_stage", default=None)
    parser.add_argument("--to-stage", dest="to_stage", default=None)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only specific stage IDs (e.g., --only 14 15 16).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable path used for subprocess stages.",
    )
    args = parser.parse_args(argv)

    stages = build_pipeline_catalog()
    ordered_ids = [stage.stage_id.lower() for stage in stages]
    by_id = {stage.stage_id.lower(): stage for stage in stages}

    if args.list:
        for stage in stages:
            print(f"{stage.stage_id:>3}  {stage.title:<28}  {stage.script_path.name}")
            print(f"     {stage.description}")
        return 0

    try:
        selected = _select_stage_ids(args, ordered_ids)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_list = [by_id[sid] for sid in ordered_ids if sid in selected]
    if not run_list:
        print("No stages selected.", file=sys.stderr)
        return 2

    for stage in run_list:
        cmd = [args.python, str(stage.script_path)]
        print(f"\n==> [{stage.stage_id}] {stage.title}")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"Stage {stage.stage_id} failed with exit code {rc}.", file=sys.stderr)
            return rc

    print(f"\nPipeline completed. Stages run: {', '.join(s.stage_id for s in run_list)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
