"""Visual encoding for roster wartime location and frontline service.

Paper figures use a deliberately minimal two-channel scheme:

* **Colour** — remained in Ukraine vs left / exile (``in_ukraine_wartime``).
  Mobilized and civilian poets who stayed are merged into the same colour.
* **Marker shape** — documented frontline / combat deployment (triangle) vs
  not at the front (square), when a second dimension is needed without adding
  a third colour category.  Mobilization alone is insufficient (e.g. rear-
  echelon spokespeople in Kyiv stay square).
"""

from __future__ import annotations

from matplotlib.lines import Line2D

REMAINED = "#2E5266"   # deep teal — in Ukraine (mobilized or civilian)
LEFT = "#8B6F47"       # muted bronze — left Ukraine / exile
UNKNOWN = "#B0A78F"    # neutral grey — location not coded

MOB_MARKER = "^"
CIV_MARKER = "s"


def _yes(value: object) -> bool:
    return str(value).strip().lower() == "yes"


def location(
    mobilized: object,
    in_ukraine_wartime: object,
    region_at_archive_freeze: object | None = None,
) -> str:
    """Return ``remained``, ``left``, or ``unknown``."""
    if _yes(in_ukraine_wartime):
        return "remained"
    if _yes(mobilized):
        return "remained"
    if str(in_ukraine_wartime).strip().lower() == "no":
        return "left"
    raf = str(region_at_archive_freeze or "").strip().lower()
    if raf == "diaspora":
        return "left"
    if raf:
        return "remained"
    return "unknown"


def location_color(
    mobilized: object,
    in_ukraine_wartime: object,
    region_at_archive_freeze: object | None = None,
) -> str:
    return {
        "remained": REMAINED,
        "left": LEFT,
        "unknown": UNKNOWN,
    }[location(mobilized, in_ukraine_wartime, region_at_archive_freeze)]


def is_mobilized(mobilized: object) -> bool:
    return _yes(mobilized)


_REAR_ECHELON_HINTS = ("spokesperson", "mod spokesperson")


def is_at_frontline(
    mobilized: object,
    in_ukraine_wartime: object,
    region_at_archive_freeze: object | None = None,
    notes: object = "",
) -> bool:
    """Return True when covariates document active frontline / combat deployment."""
    if not is_mobilized(mobilized):
        return False
    notes_l = str(notes or "").lower()
    if any(h in notes_l for h in _REAR_ECHELON_HINTS):
        return False
    if any(
        kw in notes_l
        for kw in (
            "frontline",
            "combat from",
            "combat medic",
            "air assault",
            "marine recon",
            "donetsk oblast combat",
            "mia since",
        )
    ):
        return True
    raf = str(region_at_archive_freeze or "").strip().lower()
    return raf == "east_ukraine"


def mobilization_marker(mobilized: object) -> str:
    """Backward-compatible alias; prefer :func:`frontline_marker`."""
    return MOB_MARKER if is_mobilized(mobilized) else CIV_MARKER


def frontline_marker(
    mobilized: object,
    in_ukraine_wartime: object,
    region_at_archive_freeze: object | None = None,
    notes: object = "",
) -> str:
    return MOB_MARKER if is_at_frontline(
        mobilized, in_ukraine_wartime, region_at_archive_freeze, notes
    ) else CIV_MARKER


def location_label(key: str) -> str:
    return {
        "remained": "Remained in Ukraine",
        "left": "Left Ukraine (exile)",
        "unknown": "Location unknown",
    }[key]


def legend_handles(
    *,
    include_mobilization: bool = True,
    include_frontline: bool = False,
    include_arrow: bool = False,
    ms: float = 8,
) -> list[Line2D]:
    """Standard figure legend for the colour + optional shape encoding."""
    handles: list[Line2D] = [
        Line2D([], [], marker=CIV_MARKER, linestyle="None", color=REMAINED,
               markersize=ms, label=location_label("remained")),
        Line2D([], [], marker=CIV_MARKER, linestyle="None", color=LEFT,
               markersize=ms, label=location_label("left")),
    ]
    if include_frontline:
        handles.extend([
            Line2D([], [], marker=MOB_MARKER, linestyle="None", color="#444444",
                   markersize=ms, label="at the frontline"),
            Line2D([], [], marker=CIV_MARKER, linestyle="None", color="#444444",
                   markersize=ms, label="not at the frontline"),
        ])
    elif include_mobilization:
        handles.extend([
            Line2D([], [], marker=MOB_MARKER, linestyle="None", color="#444444",
                   markersize=ms, label="mobilized"),
            Line2D([], [], marker=CIV_MARKER, linestyle="None", color="#444444",
                   markersize=ms, label="not mobilized"),
        ])
    if include_arrow:
        handles.append(
            Line2D([], [], marker=">", linestyle="-", color="#444444",
                   markersize=ms, label="dot = pre-2022 → arrow = wartime"),
        )
    return handles
