from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import ensure_dirs, minmax_series


def _safe_numeric(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _z_by_season(df: pd.DataFrame, col: str) -> pd.Series:
    x = _safe_numeric(df, col)
    mu = x.groupby(df["season"]).transform("mean")
    sd = x.groupby(df["season"]).transform("std").replace(0, np.nan)
    return ((x - mu) / sd).clip(-4, 4).fillna(0)


def build_team_features(tables: dict[str, pd.DataFrame], season_df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    """Build team-season strength features and all-time team ranking.

    The model is designed for two uses:
    1. team rankings, e.g. 2017 Warriors vs 1998 Bulls;
    2. opponent-strength support for playoff path analysis.

    It combines regular-season dominance, roster star power/depth, playoff route,
    and era-relative dominance.
    """
    out_dir = Path(out_dir)
    feature_dir = out_dir / "features"
    outputs_dir = out_dir / "outputs"
    ensure_dirs(feature_dir, outputs_dir)

    team_context = tables.get("team_context", pd.DataFrame()).copy()
    playoff_series = tables.get("playoff_series", pd.DataFrame()).copy()
    if team_context.empty:
        return pd.DataFrame()

    # Roster aggregation from player season table.
    sf = season_df.copy()
    if "team" not in sf.columns:
        sf["team"] = ""
    for c in ["season_ability_value", "season_ability_value_base", "season_legacy_value", "ws", "vorp", "bpm", "mp", "award_score_season", "all_star_score", "all_nba_score", "all_defense_score"]:
        if c not in sf.columns:
            sf[c] = np.nan
        sf[c] = pd.to_numeric(sf[c], errors="coerce")
    sf["bpm_x_mp"] = sf["bpm"].fillna(0) * sf["mp"].fillna(0).clip(lower=1)
    roster = sf.groupby(["season", "team"], dropna=False).agg(
        roster_player_count=("player_id", "nunique"),
        roster_minutes=("mp", "sum"),
        roster_ws=("ws", "sum"),
        roster_vorp=("vorp", "sum"),
        roster_award_score=("award_score_season", "sum"),
        roster_all_star_score=("all_star_score", "sum"),
        roster_all_nba_score=("all_nba_score", "sum"),
        roster_all_defense_score=("all_defense_score", "sum"),
        roster_bpm_x_mp=("bpm_x_mp", "sum"),
        roster_bpm_mp=("mp", "sum"),
    ).reset_index()
    roster["roster_bpm_weighted"] = roster["roster_bpm_x_mp"] / roster["roster_bpm_mp"].replace(0, np.nan)

    def top_n_mean(n: int, col: str, out_col: str) -> pd.DataFrame:
        tmp = sf[["season", "team", col]].dropna().sort_values(["season", "team", col], ascending=[True, True, False])
        return tmp.groupby(["season", "team"], sort=False).head(n).groupby(["season", "team"], sort=False)[col].mean().rename(out_col).reset_index()

    for n, col, out_col in [
        (1, "season_ability_value", "top1_player_ability"),
        (2, "season_ability_value", "top2_player_ability"),
        (3, "season_ability_value", "top3_player_ability"),
        (5, "season_ability_value", "top5_player_ability"),
        (3, "season_legacy_value", "top3_player_legacy"),
    ]:
        roster = roster.merge(top_n_mean(n, col, out_col), on=["season", "team"], how="left")

    df = team_context.merge(roster, on=["season", "team"], how="left")
    if not playoff_series.empty:
        df = df.merge(playoff_series, on=["season", "team"], how="left")

    for c in ["team_wins", "team_losses", "team_win_pct", "team_srs", "team_sos", "team_off_rating", "team_def_rating", "team_net_rating", "team_mov", "team_power_score_prelim", "playoff_path_difficulty", "playoff_round_score", "playoff_series_wins", "playoff_series_losses", "playoff_champion", "playoff_finals", "playoff_opponent_power_avg", "playoff_upset_wins"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Era-relative dominance.
    df["team_srs_era_z2"] = _z_by_season(df, "team_srs")
    df["team_net_rating_era_z2"] = _z_by_season(df, "team_net_rating")
    df["team_win_pct_era_z2"] = _z_by_season(df, "team_win_pct")
    df["team_off_rating_era_z2"] = _z_by_season(df, "team_off_rating")
    # Invert defensive rating: lower is better.
    tmp = df.copy()
    tmp["neg_def"] = -_safe_numeric(tmp, "team_def_rating")
    df["team_def_rating_era_z2"] = _z_by_season(tmp, "neg_def")

    df["regular_dominance_score"] = (
        (df["team_srs_era_z2"] * 0.34 + df["team_net_rating_era_z2"] * 0.24 + df["team_win_pct_era_z2"] * 0.18 + df["team_off_rating_era_z2"] * 0.10 + df["team_def_rating_era_z2"] * 0.10) * 12 + 50
    ).clip(0, 100)

    df["roster_star_score"] = (
        pd.to_numeric(df.get("top1_player_ability"), errors="coerce").fillna(50) * 0.32
        + pd.to_numeric(df.get("top2_player_ability"), errors="coerce").fillna(50) * 0.22
        + pd.to_numeric(df.get("top3_player_ability"), errors="coerce").fillna(50) * 0.18
        + minmax_series(df.get("roster_vorp", pd.Series(index=df.index))).fillna(50) * 0.12
        + minmax_series(df.get("roster_ws", pd.Series(index=df.index))).fillna(50) * 0.10
        + minmax_series(df.get("roster_award_score", pd.Series(index=df.index))).fillna(50) * 0.06
    ).clip(0, 100)
    df["roster_depth_score"] = (
        pd.to_numeric(df.get("top5_player_ability"), errors="coerce").fillna(50) * 0.38
        + minmax_series(df.get("roster_bpm_weighted", pd.Series(index=df.index))).fillna(50) * 0.22
        + minmax_series(df.get("roster_minutes", pd.Series(index=df.index))).fillna(50) * 0.10
        + minmax_series(df.get("roster_ws", pd.Series(index=df.index))).fillna(50) * 0.18
        + minmax_series(df.get("roster_all_defense_score", pd.Series(index=df.index))).fillna(50) * 0.12
    ).clip(0, 100)
    df["playoff_team_score"] = (
        df["playoff_path_difficulty"].fillna(45) * 0.32
        + minmax_series(df["playoff_round_score"].fillna(0)) * 0.20
        + minmax_series(df["playoff_series_wins"].fillna(0)) * 0.13
        + df["playoff_champion"].fillna(0) * 18
        + df["playoff_finals"].fillna(0) * 7
        + minmax_series(df["playoff_opponent_power_avg"].fillna(50)) * 0.10
        + minmax_series(df["playoff_upset_wins"].fillna(0)) * 0.05
    ).clip(0, 100)
    df["era_dominance_score"] = (
        minmax_series(df["team_srs_era_z2"] + df["team_net_rating_era_z2"] + df["team_win_pct_era_z2"]).fillna(50)
    ).clip(0, 100)
    # Final team score. Regular dominance matters most, but all-time teams need playoff proof and roster context.
    df["team_strength_score_final"] = (
        df["regular_dominance_score"] * 0.36
        + df["roster_star_score"] * 0.23
        + df["roster_depth_score"] * 0.13
        + df["playoff_team_score"] * 0.20
        + df["era_dominance_score"] * 0.08
    ).clip(0, 100)
    df["team_label"] = df["season"].astype(str) + " " + df.get("team_name", df["team"]).astype(str)
    df["team_rank_eligible"] = ((df["team_wins"].fillna(0) >= 30) | (df["playoff_series_wins"].fillna(0) > 0)).astype(int)
    df = df.sort_values("team_strength_score_final", ascending=False).reset_index(drop=True)
    eligible = df["team_rank_eligible"] == 1
    df.loc[eligible, "team_all_time_rank"] = df.loc[eligible, "team_strength_score_final"].rank(method="first", ascending=False).astype(int)

    keep_cols = [
        "team_all_time_rank", "season", "team", "team_name", "team_label", "team_strength_score_final",
        "regular_dominance_score", "roster_star_score", "roster_depth_score", "playoff_team_score", "era_dominance_score",
        "team_wins", "team_losses", "team_win_pct", "team_srs", "team_net_rating", "team_off_rating", "team_def_rating", "team_pace",
        "top1_player_ability", "top2_player_ability", "top3_player_ability", "top5_player_ability", "top3_player_legacy",
        "roster_ws", "roster_vorp", "roster_bpm_weighted", "roster_award_score", "roster_all_star_score", "roster_all_nba_score", "roster_all_defense_score",
        "playoff_series_count", "playoff_series_wins", "playoff_series_losses", "playoff_round_score", "playoff_champion", "playoff_finals",
        "playoff_path_difficulty", "playoff_opponent_power_avg", "playoff_opponent_srs_avg", "playoff_opponent_net_rating_avg", "playoff_upset_wins",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df.to_csv(feature_dir / "team_season_features.csv", index=False)
    df[keep_cols].sort_values("team_strength_score_final", ascending=False).to_csv(outputs_dir / "team_ranking_results.csv", index=False)
    # Convenience top list for app / report.
    df.loc[eligible, keep_cols].sort_values("team_strength_score_final", ascending=False).head(200).to_csv(outputs_dir / "team_ranking_all_time_top200.csv", index=False)
    return df
