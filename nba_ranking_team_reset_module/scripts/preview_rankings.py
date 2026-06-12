from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def show(path: Path, score_col: str, rank_col: str, n: int) -> None:
    df = pd.read_csv(path)
    cols = [rank_col, "player", "position_group", score_col, "peak_3_score", "playoff_score", "award_score", "career_start", "career_end"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + path.name)
    print(df[df.get("eligible", 1) == 1][cols].head(n).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()
    out = Path(args.out_dir) / "outputs"
    show(out / "ranking_legacy_overall.csv", "legacy_score_final", "legacy_rank", args.n)
    show(out / "ranking_ability_overall.csv", "ability_score_final", "ability_rank", args.n)
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        show(out / f"ranking_ability_{pos}.csv", "ability_score_final", "position_ability_rank", min(args.n, 10))


if __name__ == "__main__":
    main()
