#!/usr/bin/env python3
"""Export main.tex to Word (.docx) for Google Docs upload."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    import pypandoc
except ImportError:
    print("Install: pip install --user pypandoc-binary pymupdf", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
SRC_TEX = ROOT / "main.tex"
EXPORT_TEX = ROOT / "main_word_export.tex"
OUT_DOCX = ROOT / "main.docx"
FIG_DIR = ROOT / "figures"
FIG_PNG_DIR = ROOT / "figures_png"
BIB = ROOT / "reference.bib"

# Pandoc citeproc is case-sensitive; natbib keys in .bbl use lowercase.
CITE_KEY_FIXES = {
    "Hirsch2012the": "hirsch2012the",
}


def pdf_figures_to_png() -> None:
    if fitz is None:
        return
    FIG_PNG_DIR.mkdir(exist_ok=True)
    for pdf in sorted(FIG_DIR.glob("*.pdf")):
        png = FIG_PNG_DIR / f"{pdf.stem}.png"
        if png.exists() and png.stat().st_mtime >= pdf.stat().st_mtime:
            continue
        doc = fitz.open(pdf)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(png)


def prepare_export_tex() -> None:
    text = SRC_TEX.read_text(encoding="utf-8")
    for old, new in CITE_KEY_FIXES.items():
        text = text.replace(old, new)
    # Use PNG copies so Word/Google Docs render figures reliably.
    text = re.sub(
        r"(\\includegraphics\[[^\]]*\]\{)(fig_narr_[^}]+\.pdf)(\})",
        lambda m: f"{m.group(1)}{m.group(2).replace('.pdf', '.png')}{m.group(3)}",
        text,
    )
    text = text.replace(
        r"\graphicspath{{figures/}}",
        r"\graphicspath{{figures_png/}{figures/}}",
    )
    EXPORT_TEX.write_text(text, encoding="utf-8")


def run_pandoc() -> None:
    extra = [
        f"--bibliography={BIB}",
        "--citeproc",
        f"--resource-path=.:{FIG_DIR}:{FIG_PNG_DIR}",
        "--standalone",
    ]
    pypandoc.convert_file(
        str(EXPORT_TEX),
        "docx",
        outputfile=str(OUT_DOCX),
        extra_args=extra,
    )


def main() -> None:
    if not SRC_TEX.exists():
        raise SystemExit(f"Missing {SRC_TEX}")
    pdf_figures_to_png()
    prepare_export_tex()
    run_pandoc()
    print(f"Wrote {OUT_DOCX} ({OUT_DOCX.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
