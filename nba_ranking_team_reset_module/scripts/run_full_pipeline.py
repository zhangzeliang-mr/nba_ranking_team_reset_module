from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nba_ranker.cleaning import build_clean_tables
from nba_ranker.features import build_season_features, build_career_features
from nba_ranker.team_features import build_team_features


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA historical legacy and ability ranking pipeline")
    parser.add_argument("--base_dir", required=True, help="Directory containing nba_aba_baa_stats and other data folders")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Cleaning raw CSV files...")
    tables = build_clean_tables(args.base_dir, out_dir)
    print(f"  regular rows: {len(tables['regular'])}")
    print(f"  playoff rows: {len(tables['playoffs'])}")
    print(f"  award rows: {len(tables['awards'])}")

    print("[2/4] Building player season features...")
    season_features = build_season_features(tables, out_dir)
    print(f"  season feature rows: {len(season_features)}")

    print("[3/5] Building team strength features and team rankings...")
    team_features = build_team_features(tables, season_features, out_dir)
    print(f"  team feature rows: {len(team_features)}")

    print("[4/5] Building player career features...")
    career_features = build_career_features(season_features, tables, out_dir)
    print(f"  career feature rows: {len(career_features)}")

    print("[5/5] Training ML + deep models and generating rankings...")
    # Run model training in a fresh Python process. This avoids numerical-library thread conflicts
    # that may occur when sklearn is imported after heavy pandas groupby work in the same process.
    import subprocess
    train_script = ROOT / "scripts" / "train_from_features.py"
    features_csv = out_dir / "features" / "player_career_features.csv"
    subprocess.run([sys.executable, str(train_script), "--features_csv", str(features_csv), "--out_dir", str(out_dir)], check=True)
    print("Done.")
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
