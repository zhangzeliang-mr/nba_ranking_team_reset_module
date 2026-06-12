from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview all-time team season rankings")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--team", default=None, help="Optional team abbreviation, e.g. GSW or CHI")
    parser.add_argument("--season", type=int, default=None, help="Optional season, e.g. 2017")
    args = parser.parse_args()
    f = Path(args.out_dir) / "outputs" / "team_ranking_results.csv"
    df = pd.read_csv(f)
    if args.team:
        df = df[df["team"].astype(str).str.upper() == args.team.upper()]
    if args.season:
        df = df[df["season"] == args.season]
    cols = ["team_all_time_rank", "season", "team", "team_name", "team_strength_score_final", "regular_dominance_score", "roster_star_score", "roster_depth_score", "playoff_team_score"]
    print(df[cols].head(args.n).to_string(index=False))


if __name__ == "__main__":
    main()
