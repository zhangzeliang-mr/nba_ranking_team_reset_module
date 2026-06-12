from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

URL = "https://www.basketball-reference.com/playoffs/series.html"


def _clean_col(c) -> str:
    if isinstance(c, tuple):
        c = "_".join([str(x) for x in c if str(x).strip() and str(x) != "nan"])
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
    return c


def scrape_bref_series(url: str = URL) -> pd.DataFrame:
    """Scrape Basketball-Reference NBA/ABA playoff series history.

    Output schema:
        season, league, series_round, winner_team, winner_wins,
        loser_team, loser_wins, favorite, underdog

    Usage:
        python scripts/crawl_playoff_series.py --out_file D:/nba_data/playoff_series.csv

    Note: Basketball-Reference can rate-limit/deny automated requests. If that happens,
    open the page in a browser, save the table as CSV manually, and place it at one of:
        <base_dir>/playoff_series.csv
        <base_dir>/nba_playoff_series/playoff_series.csv
        <base_dir>/raw/playoff_series.csv
    """
    tables = pd.read_html(url)
    if not tables:
        raise RuntimeError("No tables found on Basketball-Reference playoff series page.")
    # Usually the largest table is the Playoffs Series Table.
    df = max(tables, key=len).copy()
    df.columns = [_clean_col(c) for c in df.columns]
    # Try to normalize common Basketball-Reference column variants.
    rename_candidates = {
        "yr": "season",
        "year": "season",
        "lg": "league",
        "winner_team": "winner_team",
        "winner_w": "winner_wins",
        "loser_team": "loser_team",
        "loser_w": "loser_wins",
        "series": "series_round",
    }
    for k, v in list(rename_candidates.items()):
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})
    # MultiIndex flattening may create names like winner_team, winner_w, loser_team, loser_w.
    cols = list(df.columns)
    if "winner_team" not in cols:
        team_cols = [c for c in cols if c.endswith("team") or c == "team"]
        if len(team_cols) >= 2:
            df = df.rename(columns={team_cols[0]: "winner_team", team_cols[1]: "loser_team"})
    if "winner_wins" not in df.columns:
        wcols = [c for c in cols if c.endswith("_w") or c == "w"]
        if len(wcols) >= 2:
            df = df.rename(columns={wcols[0]: "winner_wins", wcols[1]: "loser_wins"})
    required = ["season", "winner_team", "loser_team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Could not normalize playoff series table. Missing: {missing}. Columns: {list(df.columns)}")
    if "series_round" not in df.columns:
        # Some exports put round in a column containing 'series'.
        series_cols = [c for c in df.columns if "series" in c]
        if series_cols:
            df = df.rename(columns={series_cols[0]: "series_round"})
        else:
            df["series_round"] = "Unknown"
    keep = [c for c in ["season", "league", "series_round", "winner_team", "winner_wins", "loser_team", "loser_wins", "favorite", "underdog"] if c in df.columns]
    out = df[keep].copy()
    out = out[out["season"].astype(str).str.extract(r"(\d{4})", expand=False).notna()]
    out["season"] = out["season"].astype(str).str.extract(r"(\d{4})", expand=False).astype(int)
    for c in ["winner_wins", "loser_wins"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.drop_duplicates()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = scrape_bref_series(args.url)
    df.to_csv(out_file, index=False)
    print(f"Saved {len(df)} playoff series rows to {out_file}")


if __name__ == "__main__":
    main()
