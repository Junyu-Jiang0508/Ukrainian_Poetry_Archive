#!/usr/bin/env python3
"""
Extract APA-style citation lines from Compass markdown artifacts and emit merged BibTeX.
Heuristic parser: prefers DOI for dedup; fills journal/booktitle/pages when parseable.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

DOI_RE = re.compile(r"https?://doi\.org/((?:10\.)[^\s\]]+)", re.I)
ARXIV_RE = re.compile(r"arXiv:(\d{4}\.\d{5})", re.I)
HDL_RE = re.compile(r"https?://hdl\.handle\.net/([^\s\]]+)", re.I)

# **Author (YYYY).** rest  (thread entries)
CITE_LINE_A = re.compile(
    r"^\*\*(.+?)\s*\((\d{4}[a-z]?)\)\.\*\*\s*(.*)$"
)
# **N.** Author (YYYY). rest  — note Markdown is ** + N. + ** + space + Author...
CITE_LINE_B = re.compile(
    r"^\*\*\d+\.\*\*\s*(.+?)\s*\((\d{4}[a-z]?)\)\.\s*(.*)$"
)

# Optional lines: *Optional supplementary*: Author (Y). ...
OPT_LINE = re.compile(
    r"^\*Optional supplementary\*:\s*(.+?)\s*\((\d{4}[a-z]?)\)\.\s*(.*)$",
    re.I,
)


def slug(s: str, max_len: int = 32) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if ch.isalnum())
    return (s[:max_len] or "x").lower()


def first_author_last(authors: str) -> str:
    authors = authors.strip()
    # "van Dijk, T. A." -> vandijk ; "Brubaker, R." -> brubaker
    first = authors.split("&")[0].split(",")[0].strip()
    first = re.sub(r"\b(van|de|von|del|la|le)\s+", "", first, flags=re.I)
    return slug(re.sub(r"[^A-Za-z]", "", first), 20)


def bib_escape(s: str) -> str:
    s = s.replace("🦜", "").strip()
    return s.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}")


def strip_md_italics(s: str) -> str:
    s = re.sub(r"\*+([^*]+)\*+", r"\1", s)
    return s.strip()


def parse_authors_bibtex(authors: str) -> str:
    """Convert APA-style author list to BibTeX 'and' list."""
    a = authors.replace("…", "...").strip()
    a = re.sub(r"\s*\(2nd ed\.,.*?\)\s*$", "", a, flags=re.I | re.S)
    a = re.sub(r",\s*&\s*", " and ", a)
    names: list[str] = []
    for outer in re.split(r"\s+and\s+", a):
        outer = outer.strip().rstrip(",").strip()
        if not outer:
            continue
        # Split "Last, F., Last2, F2." on commas that separate author blocks
        sub = re.split(r"(?<=\.)\s*,\s*(?=[A-ZÀ-ÖØ-ſ\u0100-\u024f][A-Za-zÀ-ÖØ-öø-ÿ\u0100-\u024f\-]*,\s)", outer)
        for p in sub:
            p = p.strip().rstrip(",").strip()
            if not p:
                continue
            names.append(p)
    bib_names: list[str] = []
    for p in names:
        if "," in p:
            last, first = p.split(",", 1)
            bib_names.append(f"{last.strip()}, {first.strip()}")
        else:
            bib_names.append(p)
    return " and ".join(bib_names)


def infer_entry(rest: str, doi: Optional[str]) -> tuple[str, dict]:
    """Return (entry_type, fields)."""
    rest_clean = strip_md_italics(rest)
    r = rest_clean
    fields: dict[str, str] = {}
    if doi:
        fields["doi"] = doi

    # Article: Title. *Journal Name, vol*(issue), pages.  (common in both MD sources)
    jm_star = re.search(
        r"^(.+?)\.\s*\*(.+?),\s*(\d+)\*\(([^)]*)\)\s*,\s*([\d–\-]+)",
        rest.strip(),
    )
    if jm_star:
        fields["title"] = bib_escape(jm_star.group(1).strip())
        fields["journal"] = bib_escape(jm_star.group(2).strip())
        fields["volume"] = jm_star.group(3).strip()
        if jm_star.group(4).strip():
            fields["number"] = jm_star.group(4).strip()
        fields["pages"] = jm_star.group(5).replace("-", "--")
        return "article", fields

    # Article: Title. *Journal*, (issue), pages.
    jm_paren = re.search(
        r"^(.+?)\.\s*\*([^*]+)\*\s*,\s*\(([^)]*)\)\s*,\s*([\d–\-]+)",
        rest.strip(),
    )
    if jm_paren:
        fields["title"] = bib_escape(jm_paren.group(1).strip())
        fields["journal"] = bib_escape(jm_paren.group(2).strip())
        fields["number"] = jm_paren.group(3).strip()
        fields["pages"] = jm_paren.group(4).replace("-", "--")
        return "article", fields

    # Book: *Title*. Publisher... (use raw rest so italics remain)
    bk = re.match(r"^\*([^*]+)\*\s*\.\s*(.+)$", rest.strip())
    if bk:
        fields["title"] = bib_escape(bk.group(1).strip())
        tail = bk.group(2).strip()
        tail = re.sub(r"\([^)]*Series[^)]*\)", "", tail, flags=re.I)
        tail = re.sub(r"https?://doi\.org/\S+", "", tail).strip().rstrip(".")
        if tail and not tail.lower().startswith("http"):
            pub = tail.split("http")[0].strip().rstrip(".")
            pub = re.sub(r"\s+[—–]\s+.*$", "", pub)
            fields["publisher"] = bib_escape(pub)
        return "book", fields

    low = r.lower()
    if "doctoral dissertation" in low or "[doctoral dissertation" in low:
        m_uni = re.search(r"\[Doctoral dissertation,\s*([^\]]+)\]", r, re.I)
        thesis_type = "phdthesis"
        fields["title"] = bib_escape(strip_md_italics(re.sub(r"\[Doctoral dissertation.*$", "", r).strip()))
        if m_uni:
            fields["school"] = m_uni.group(1).strip()
        return thesis_type, fields

    if re.search(r"\barXiv:", r, re.I) or "arXiv preprint" in low:
        fields["title"] = bib_escape(re.sub(r"\s*\(arXiv:.*?\)\s*.*$", "", r).strip())
        m = ARXIV_RE.search(r)
        if m:
            fields["journal"] = "arXiv"
            fields["volume"] = m.group(1)
        return "article", fields

    if re.search(r"In\s+.*Proceedings of\s+", r, re.I) or re.search(
        r"In\s+\*Proceedings of", rest, re.I
    ):
        src = rest.strip()
        im = re.split(r"\s+In\s+\*Proceedings of", src, maxsplit=1, flags=re.I)
        if im and im[0].strip():
            fields["title"] = bib_escape(
                re.sub(r"\s*\[(?:FOUNDATIONAL|CLASSIC)[^\]]*\]\s*$", "", strip_md_italics(im[0].strip()), flags=re.I).strip()
            )
        if len(im) == 2:
            tailp = "*Proceedings of" + im[1]
            bm = re.search(r"\*([^*]+)\*", tailp)
            if bm:
                fields["booktitle"] = bib_escape(bm.group(1).strip())
            ppm = re.search(r"\(pp\.\s*([\d–\-]+)\)", tailp)
            if ppm:
                fields["pages"] = ppm.group(1).replace("-", "--").replace("–", "--")
            org = re.search(r"\)\s*\.\s*([A-Z][A-Za-z]+)\.", tailp)
            if org:
                fields["publisher"] = bib_escape(org.group(1).strip())
        return "inproceedings", fields

    if re.search(r"\bin\s+[A-Z]", r) and "pp." in r and not re.search(
        r"^\s*[A-Z][^.]+\.\s*\*", r
    ):
        # book chapter: "Title. In Editor (Ed.), *Book* (pp. x-y). Publisher."
        return "incollection", fields

    # Journal: *Name, vol*(issue), pages
    jm = re.search(
        r"^([^.]+)\.\s*\*([^*]+)\*,\s*(\d+)\s*\(\s*([^)]*)\s*\)\s*,\s*([\d–\-]+)",
        r,
    )
    if jm:
        fields["title"] = bib_escape(jm.group(1).strip())
        fields["journal"] = bib_escape(jm.group(2).strip())
        fields["volume"] = jm.group(3).strip()
        num = jm.group(4).strip()
        if num:
            fields["number"] = num
        fields["pages"] = jm.group(5).replace("-", "--")
        tail = r[jm.end() :].strip()
        pub = re.sub(r"^[\s.]*", "", tail)
        if pub and not pub.startswith("http"):
            fields["publisher"] = bib_escape(pub.split("http")[0].strip().rstrip("."))
        return "article", fields

    # Journal without capturing via simpler heuristic: *Journal*
    jm2 = re.search(r"\*([^*]+)\*,\s*(\d+)\s*\(", r)
    if jm2 and "http" not in jm2.group(1) and len(jm2.group(1)) < 120:
        # title = text before first * if starts with quote or capital sentence
        title_m = re.match(r"^“?([^*]+?)\.\s*\*", r)
        if title_m:
            fields["title"] = bib_escape(title_m.group(1).strip().strip("“”"))
        else:
            fields["title"] = bib_escape(r.split("*", 1)[0].strip())
        fields["journal"] = bib_escape(jm2.group(1).strip())
        vm = re.search(r",\s*(\d+)\s*\(\s*([^)]*)\)\s*,\s*([\d–\-]+)", r)
        if vm:
            fields["volume"] = vm.group(1)
            if vm.group(2).strip():
                fields["number"] = vm.group(2).strip()
            fields["pages"] = vm.group(3).replace("-", "--")
        return "article", fields

    # Book: *Title* ... Publisher
    bm = re.match(r"^\*([^*]+)\*\s*(.*)$", r.strip())
    if bm:
        fields["title"] = bib_escape(bm.group(1).strip())
        tail = bm.group(2).strip()
        # series in parens
        tail = re.sub(r"\([^)]*Series[^)]*\)", "", tail, flags=re.I)
        tail = tail.rstrip(".")
        if tail and not tail.startswith("http"):
            fields["publisher"] = bib_escape(tail.split("https://doi.org")[0].strip())
        return "book", fields

    # Fallback misc
    fields["title"] = bib_escape(r[:300])
    fields["howpublished"] = "Parsed from Compass markdown; verify full metadata."
    return "misc", fields


def extract_doi(rest: str) -> Optional[str]:
    m = DOI_RE.search(rest)
    return m.group(1).rstrip(").,;") if m else None


def extract_hdl(rest: str) -> Optional[str]:
    m = HDL_RE.search(rest)
    return m.group(1).rstrip(").,;") if m else None


def cite_key(authors: str, year: str, title: str) -> str:
    la = first_author_last(authors)
    tt = slug(re.sub(r"[^A-Za-z0-9]+", "", title.split()[0] if title.split() else "t"), 18)
    return f"{la}{year}{tt}"


def merge_fields(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k not in out or len(str(v)) > len(str(out.get(k, ""))):
            out[k] = v
    return out


def parse_line(authors: str, year: str, rest: str) -> Optional[tuple[str, str, dict]]:
    authors = re.sub(r"^\*+\s*", "", authors.strip())
    authors_bib = parse_authors_bibtex(authors)
    doi = extract_doi(rest)
    hdl = extract_hdl(rest)
    etype, fields = infer_entry(rest, doi)
    title_for_key = fields.get("title", rest[:40])
    key = cite_key(authors, year, strip_md_italics(title_for_key))

    entry = dict(fields)
    entry["author"] = authors_bib
    entry["year"] = year
    if hdl and "url" not in entry:
        entry["url"] = f"https://hdl.handle.net/{hdl}"

    # Fill missing title for books detected
    if "title" not in entry or entry["title"] == "":
        m = re.match(r"^\*([^*]+)\*", rest.strip())
        if m:
            entry["title"] = bib_escape(m.group(1).strip())

    return key, etype, entry


def dedup_key(doi: Optional[str], author: str, year: str, title: str) -> str:
    if doi:
        return doi.lower()
    t = strip_md_italics(title)[:60].lower()
    return f"{first_author_last(author)}|{year}|{t}"


def emit_bib(key: str, etype: str, fields: dict) -> str:
    lines = [f"@{etype}{{{key},"]
    order = ["author", "year", "title", "booktitle", "editor", "series", "volume", "number", "pages", "journal", "publisher", "organization", "address", "school", "doi", "url", "howpublished", "note"]
    done = set()
    for k in order:
        if k in fields and fields[k]:
            lines.append(f"  {k} = {{{fields[k]}}},")
            done.add(k)
    for k in sorted(fields):
        if k not in done and fields[k]:
            lines.append(f"  {k} = {{{fields[k]}}},")
    lines.append("}")
    return "\n".join(lines)


def extract_from_text(text: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        m = CITE_LINE_A.match(line) or CITE_LINE_B.match(line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3)))
            continue
        m2 = OPT_LINE.match(line)
        if m2:
            rows.append((m2.group(1), m2.group(2), m2.group(3)))
            # Body 5 optional bundles two works on one line
            rest3 = m2.group(3)
            if " Andryczyk," in rest3:
                tail = "Andryczyk," + rest3.split(" Andryczyk,", 1)[1]
                m3 = re.match(
                    r"^Andryczyk,\s*M\.\s*\(Ed\.\)\.\s*\((\d{4}[a-z]?)\)\.\s*(.*)$",
                    tail.strip(),
                )
                if m3:
                    rows.append(("Andryczyk, M.", m3.group(1), m3.group(2)))
            continue
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "docs" / "compass_artifact_wf-50ccf845-d725-46b9-8e40-f57665758acc_text_markdown.md",
        root / "docs" / "compass_artifact_wf-b935a608-ab45-438e-abf2-000aff0e7ae1_text_markdown.md",
    ]
    all_rows: list[tuple[str, str, str, str]] = []  # file, author, year, rest
    for p in paths:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        for a, y, r in extract_from_text(txt):
            all_rows.append((p.name, a, y, r))

    by_dedup: dict[str, tuple[str, str, dict]] = {}
    key_collisions: dict[str, int] = {}

    for _fn, authors, year, rest in all_rows:
        parsed = parse_line(authors, year, rest)
        if not parsed:
            continue
        key, etype, fields = parsed
        dk = dedup_key(fields.get("doi"), authors, year, fields.get("title", rest))
        if dk in by_dedup:
            old_key, old_type, old_fields = by_dedup[dk]
            merged = merge_fields(old_fields, fields)
            by_dedup[dk] = (old_key, old_type, merged)
            continue
        # resolve key uniqueness
        base_key = key
        if key in key_collisions:
            key_collisions[key] += 1
            key = f"{base_key}{key_collisions[key]}"
        else:
            key_collisions[key] = 0
        by_dedup[dk] = (key, etype, fields)

    header = """% Merged bibliography from Compass markdown artifacts:
% - compass_artifact_wf-50ccf845-d725-46b9-8e40-f57665758acc_text_markdown.md
% - compass_artifact_wf-b935a608-ab45-438e-abf2-000aff0e7ae1_text_markdown.md
% Generated by scripts/md_to_bib.py — verify journal volumes, pages, and editions against primary sources.
%
"""
    chunks = [header]
    # stable sort by author+year
    items = sorted(by_dedup.values(), key=lambda t: (t[2].get("author", ""), t[2].get("year", ""), t[0]))
    for key, etype, fields in items:
        chunks.append(emit_bib(key, etype, fields))
        chunks.append("")

    out = root / "docs" / "merged_compass_bibliography.bib"
    out.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {len(items)} entries to {out}")


if __name__ == "__main__":
    main()
