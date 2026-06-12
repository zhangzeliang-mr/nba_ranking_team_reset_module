from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nba_ranker.models import train_and_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ranking models from player_career_features.csv")
    parser.add_argument("--features_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    career_features = pd.read_csv(args.features_csv)
    ranking = train_and_rank(career_features, args.out_dir)
    print(f"  ranking rows: {len(ranking)}")


if __name__ == "__main__":
    main()
