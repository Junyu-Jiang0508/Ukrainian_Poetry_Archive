"""Adaptive calendar binning for time-indexed token tables (used by breakpoint regression)."""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_POEMS_PER_INTERVAL = 30
INITIAL_MONTHS = 2
TARGET_POEMS = 50
                                                                                           
TARGET_POEMS_PER_BALANCED_BIN = 200


def balanced_temporal_binning(
    df: pd.DataFrame,
    date_col: str = "date",
    id_col: str = "poem_id",
    *,
    target_poems_per_bin: int | None = None,
    min_poems: int | None = None,
    weight_col: str | None = None,
    balance_axis: str = "poems",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time bins with similar poem or stanza counts per bin (chronological, non-overlapping)."""
    min_poems = min_poems or MIN_POEMS_PER_INTERVAL
    tpb = int(target_poems_per_bin or TARGET_POEMS_PER_BALANCED_BIN)
    tpb = max(1, tpb)

    sub = df.copy()
    sub["_date"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=["_date"])
    if sub.empty:
        empty_iv = pd.DataFrame(
            columns=["interval_id", "interval_label", "n_poems", "start_date", "end_date"]
        )
        return sub.iloc[0:0].copy(), empty_iv

    use_weights = balance_axis == "stanzas" and weight_col and weight_col in sub.columns
    if balance_axis == "stanzas" and not use_weights:
        balance_axis = "poems"

    agg_cols: dict = {"_date": ("_date", "min")}
    if use_weights:
        agg_cols[weight_col] = (weight_col, "sum")
    one = sub.groupby(id_col, as_index=False).agg(**agg_cols)
    one = one.sort_values("_date", kind="mergesort").reset_index(drop=True)
    n = len(one)
    if n == 0:
        empty_iv = pd.DataFrame(
            columns=["interval_id", "interval_label", "n_poems", "start_date", "end_date"]
        )
        return sub.iloc[0:0].copy(), empty_iv

    n_bins = max(1, (n + tpb - 1) // tpb)

    if balance_axis == "stanzas" and use_weights:
        w = np.maximum(one[weight_col].to_numpy(dtype=float), 1.0)
        c = np.cumsum(w)
        total_w = float(c[-1])
        if n_bins == 1:
            bin_rank = np.zeros(n, dtype=np.int32)
        else:
            edges = np.linspace(0.0, total_w, n_bins + 1)
            bin_rank = np.searchsorted(edges[1:], c, side="left").astype(np.int32)
            bin_rank = np.clip(bin_rank, 0, n_bins - 1)
        pre = bin_rank
    else:
        q, r = divmod(n, n_bins)
        sizes = [q + (1 if j < r else 0) for j in range(n_bins)]
        pre = np.empty(n, dtype=np.int32)
        pos = 0
        for j, sz in enumerate(sizes):
            pre[pos : pos + sz] = j
            pos += sz

    ranges: list[list[int]] = []
    b0 = int(pre[0])
    start = 0
    for i in range(1, n):
        if int(pre[i]) != b0:
            ranges.append(list(range(start, i)))
            start = i
            b0 = int(pre[i])
    ranges.append(list(range(start, n)))

    i_m = 0
    while i_m < len(ranges):
        if len(ranges[i_m]) < min_poems and len(ranges) > 1:
            if i_m < len(ranges) - 1:
                ranges[i_m] = ranges[i_m] + ranges[i_m + 1]
                ranges.pop(i_m + 1)
            elif i_m > 0:
                ranges[i_m - 1] = ranges[i_m - 1] + ranges[i_m]
                ranges.pop(i_m)
                i_m -= 1
        else:
            i_m += 1

    intervals: list[dict] = []
    id_to_iv: dict[str, int] = {}
    for j, idxs in enumerate(ranges, start=1):
        sl = one.iloc[idxs]
        rec: dict = {
            "interval_id": j,
            "start_date": sl["_date"].min(),
            "end_date": sl["_date"].max(),
            "n_poems": int(len(sl)),
        }
        if use_weights:
            rec["n_weight"] = float(sl[weight_col].sum())
        intervals.append(rec)
        for pid in sl[id_col].astype(str):
            id_to_iv[str(pid)] = j

    interval_df = pd.DataFrame(intervals)
    interval_df["interval_label"] = interval_df.apply(
        lambda r: f"{r['start_date'].strftime('%Y-%m')} to {r['end_date'].strftime('%Y-%m')}",
        axis=1,
    )
    if use_weights and "n_weight" in interval_df.columns:
        interval_df = interval_df.rename(columns={"n_weight": "n_stanzas_bin_definition"})

    assign = one[[id_col]].copy()
    assign[id_col] = assign[id_col].astype(str)
    assign["interval_id"] = assign[id_col].map(id_to_iv).astype(int)

    out = sub.copy()
    out[id_col] = out[id_col].astype(str)
    out = out.merge(assign[[id_col, "interval_id"]], on=id_col, how="inner")
    merge_cols = ["interval_id", "interval_label", "n_poems", "start_date", "end_date"]
    if "n_stanzas_bin_definition" in interval_df.columns:
        merge_cols.append("n_stanzas_bin_definition")
    out = out.merge(interval_df[merge_cols], on="interval_id", how="left")
    return out, interval_df


def adaptive_binning(
    df: pd.DataFrame,
    date_col: str = "date",
    id_col: str = "ID",
    min_poems: int | None = None,
    initial_months: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_poems = min_poems or MIN_POEMS_PER_INTERVAL
    initial_months = initial_months or INITIAL_MONTHS

    df = df.copy()
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_date"])
    df["_period"] = df["_date"].dt.to_period(f"{initial_months}M")

    poem_counts = (
        df.groupby("_period")
        .agg(
            n_poems=(id_col, "nunique"),
            start_date=("_date", "min"),
            end_date=("_date", "max"),
        )
        .reset_index()
        .sort_values("_period")
    )

    intervals = poem_counts.to_dict("records")
    i = 0
    while i < len(intervals):
        if intervals[i]["n_poems"] < min_poems and len(intervals) > 1:
            if i < len(intervals) - 1:
                intervals[i]["end_date"] = intervals[i + 1]["end_date"]
                intervals[i]["n_poems"] += intervals[i + 1]["n_poems"]
                intervals.pop(i + 1)
            elif i > 0:
                intervals[i - 1]["end_date"] = intervals[i]["end_date"]
                intervals[i - 1]["n_poems"] += intervals[i]["n_poems"]
                intervals.pop(i)
                i -= 1
        else:
            i += 1

    interval_df = pd.DataFrame(intervals)
    interval_df["interval_id"] = range(1, len(interval_df) + 1)
    interval_df["interval_label"] = interval_df.apply(
        lambda r: f"{r['start_date'].strftime('%Y-%m')} to {r['end_date'].strftime('%Y-%m')}",
        axis=1,
    )

    def _assign_interval(row):
        for _, r in interval_df.iterrows():
            if r["start_date"] <= row["_date"] <= r["end_date"]:
                return r["interval_id"]
        return np.nan

    df["interval_id"] = df.apply(_assign_interval, axis=1)
    df = df.dropna(subset=["interval_id"])
    df["interval_id"] = df["interval_id"].astype(int)

    return df.merge(
        interval_df[["interval_id", "interval_label", "n_poems", "start_date", "end_date"]],
        on="interval_id",
    ), interval_df
