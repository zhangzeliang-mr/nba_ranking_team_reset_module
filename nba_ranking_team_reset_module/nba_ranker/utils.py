from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

TRADED_TEAM_MARKERS = {"TOT", "2TM", "3TM", "4TM", "5TM"}

POSITION_PRIORITY = ["PG", "SG", "SF", "PF", "C"]


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, **kwargs)


def safe_numeric(df: pd.DataFrame, skip: Iterable[str] = ()) -> pd.DataFrame:
    skip = set(skip)
    out = df.copy()
    for c in out.columns:
        if c not in skip:
            out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def season_to_int(x) -> Optional[int]:
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r"(\d{4})", s)
    if not m:
        return np.nan
    year = int(m.group(1))
    # Basketball Reference style: 2025 means 2024-25 ending year.
    # Already numeric season in this dataset usually means ending year.
    if "-" in s and len(s) >= 7:
        # 2017-18 -> 2018
        p = re.findall(r"\d+", s)
        if len(p) >= 2:
            start = int(p[0])
            end2 = int(p[1])
            return (start // 100) * 100 + end2 if end2 > 40 else (start // 100) * 100 + 100 + end2
    return year


def normalize_name(s) -> str:
    if pd.isna(s):
        return ""
    return (
        str(s)
        .replace("*", "")
        .replace("\xa0", " ")
        .strip()
        .lower()
    )


def primary_position(pos: str) -> str:
    if pd.isna(pos):
        return "UNK"
    s = str(pos).upper().replace("-", "/")
    parts = re.split(r"[/, ]+", s)
    for p in POSITION_PRIORITY:
        if p in parts or p in s:
            return p
    # common single letters
    if "G" == s:
        return "SG"
    if "F" == s:
        return "SF"
    return "UNK"


def minmax_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(50.0, index=s.index)
    lo = x.quantile(lower_q)
    hi = x.quantile(upper_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(50.0, index=s.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo) * 100).fillna(50.0)


def zscore_by_group(df: pd.DataFrame, group_col: str, cols: list[str], suffix: str = "_era_z") -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce")
        mu = x.groupby(out[group_col]).transform("mean")
        sd = x.groupby(out[group_col]).transform("std").replace(0, np.nan)
        out[col + suffix] = ((x - mu) / sd).clip(-4, 4).fillna(0.0)
    return out


def robust_score_from_z(df: pd.DataFrame, weighted_cols: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    total_w = 0.0
    for col, w in weighted_cols.items():
        if col in df.columns:
            score += pd.to_numeric(df[col], errors="coerce").fillna(0.0) * w
            total_w += abs(w)
    if total_w == 0:
        return pd.Series(50.0, index=df.index)
    # Convert z-like score to 0-100 with soft clipping.
    z = (score / total_w).clip(-3, 3)
    return (50 + z * 16.6667).clip(0, 100)


def choose_one_row_per_player_season(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate rows where traded players appear multiple times.

    Preference:
    1. total row marker: TOT/2TM/3TM/4TM
    2. row with highest minutes
    """
    if df.empty:
        return df
    data = df.copy()
    if "team" not in data.columns:
        return data.drop_duplicates(["player_id", "season"], keep="first")
    data["_team_pref"] = data["team"].astype(str).str.upper().isin(TRADED_TEAM_MARKERS).astype(int)
    mp_col = "mp" if "mp" in data.columns else None
    if mp_col:
        data["_mp_pref"] = pd.to_numeric(data[mp_col], errors="coerce").fillna(0)
    else:
        data["_mp_pref"] = 0
    data = data.sort_values(["player_id", "season", "_team_pref", "_mp_pref"], ascending=[True, True, False, False])
    data = data.drop_duplicates(["player_id", "season"], keep="first")
    return data.drop(columns=["_team_pref", "_mp_pref"], errors="ignore")


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def find_base_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    if (base / "nba_aba_baa_stats").exists():
        return base
    if (base / "nba_data" / "nba_aba_baa_stats").exists():
        return base / "nba_data"
    raise FileNotFoundError("Cannot find nba_aba_baa_stats under base_dir")
