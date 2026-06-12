from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import ensure_dirs, minmax_series, robust_score_from_z, zscore_by_group

TRADED_TEAM_MARKERS = {"TOT", "2TM", "3TM", "4TM", "5TM"}


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _load_team_context_from_base(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a season-team context table if cleaning.py provided it."""
    team_context = tables.get("team_context")
    if team_context is None or team_context.empty:
        return pd.DataFrame(columns=["season", "team"])
    return team_context.copy()


def _add_team_and_environment_context(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add teammate strength, team strength and opponent/schedule strength features.

    The current dataset has strong player-season and team-season tables, but it does not
    contain a complete playoff series opponent table. Therefore opponent strength is first
    implemented as a schedule-strength/team-environment proxy using SOS and opponent/defense
    columns. The feature names are intentionally stable so that a future playoff_series table
    can be merged without changing downstream model code.
    """
    out = df.copy()
    for c in ["ws", "vorp", "bpm", "mp", "award_score_season", "all_star_score", "all_nba_score"]:
        out[c] = _safe_num(out, c, 0.0)

    # Teammate strength from the same team-season, excluding the player himself.
    valid_team = ~out.get("team", pd.Series("", index=out.index)).astype(str).str.upper().isin(TRADED_TEAM_MARKERS)
    team_key = ["season", "team"]
    base = out.loc[valid_team, team_key + ["player_id", "ws", "vorp", "bpm", "mp", "award_score_season", "all_star_score", "all_nba_score"]].copy()
    if not base.empty:
        base["bpm_x_mp"] = base["bpm"].fillna(0) * base["mp"].fillna(0).clip(lower=1)
        team_totals = base.groupby(team_key, dropna=False).agg(
            team_player_count=("player_id", "nunique"),
            team_ws_sum=("ws", "sum"),
            team_vorp_sum=("vorp", "sum"),
            team_mp_sum=("mp", "sum"),
            team_bpm_x_mp_sum=("bpm_x_mp", "sum"),
            team_award_score_sum=("award_score_season", "sum"),
            team_all_star_score_sum=("all_star_score", "sum"),
            team_all_nba_score_sum=("all_nba_score", "sum"),
        ).reset_index()
        out = out.merge(team_totals, on=team_key, how="left")
    else:
        for c in ["team_player_count", "team_ws_sum", "team_vorp_sum", "team_mp_sum", "team_bpm_x_mp_sum", "team_award_score_sum", "team_all_star_score_sum", "team_all_nba_score_sum"]:
            out[c] = np.nan

    out["teammate_ws_sum"] = _safe_num(out, "team_ws_sum", np.nan) - out["ws"].fillna(0)
    out["teammate_vorp_sum"] = _safe_num(out, "team_vorp_sum", np.nan) - out["vorp"].fillna(0)
    out["teammate_minutes_available"] = _safe_num(out, "team_mp_sum", np.nan) - out["mp"].fillna(0)
    out["teammate_award_score_sum"] = _safe_num(out, "team_award_score_sum", np.nan) - out["award_score_season"].fillna(0)
    out["teammate_all_star_score_sum"] = _safe_num(out, "team_all_star_score_sum", np.nan) - out["all_star_score"].fillna(0)
    out["teammate_all_nba_score_sum"] = _safe_num(out, "team_all_nba_score_sum", np.nan) - out["all_nba_score"].fillna(0)
    denom = out["teammate_minutes_available"].replace(0, np.nan)
    out["teammate_bpm_weighted_avg"] = (_safe_num(out, "team_bpm_x_mp_sum", np.nan) - out["bpm"].fillna(0) * out["mp"].fillna(0).clip(lower=1)) / denom
    out["teammate_context_missing"] = (
        out.get("team", pd.Series("", index=out.index)).astype(str).str.upper().isin(TRADED_TEAM_MARKERS)
        | out["team_player_count"].isna()
        | (out["team_player_count"].fillna(0) <= 1)
    ).astype(int)

    teammate_components = [
        minmax_series(out["teammate_ws_sum"]),
        minmax_series(out["teammate_vorp_sum"]),
        minmax_series(out["teammate_bpm_weighted_avg"]),
        minmax_series(out["teammate_award_score_sum"]),
        minmax_series(out["teammate_minutes_available"]),
    ]
    out["teammate_strength_score"] = (
        teammate_components[0] * 0.30
        + teammate_components[1] * 0.25
        + teammate_components[2] * 0.20
        + teammate_components[3] * 0.15
        + teammate_components[4] * 0.10
    ).clip(0, 100)
    out.loc[out["teammate_context_missing"] == 1, "teammate_strength_score"] = 50.0

    # Team and opponent/schedule context from team summaries.
    team_context = _load_team_context_from_base(tables)
    if not team_context.empty:
        out = out.merge(team_context, on=["season", "team"], how="left")

    playoff_series = tables.get("playoff_series")
    if playoff_series is not None and not playoff_series.empty:
        out = out.merge(playoff_series, on=["season", "team"], how="left")

    for c in ["team_win_pct", "team_srs", "team_net_rating", "team_off_rating", "team_def_rating", "team_pace", "team_sos", "team_playoffs",
              "playoff_path_difficulty", "playoff_opponent_strength_score", "playoff_opponent_srs_avg", "playoff_opponent_net_rating_avg", "playoff_round_score"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Team strength: how good the player environment/team was.
    # A strong team is helpful context but can reduce single-player credit in ability scoring.
    team_strength = (
        minmax_series(out["team_win_pct"]) * 0.30
        + minmax_series(out["team_srs"]) * 0.30
        + minmax_series(out["team_net_rating"]) * 0.25
        + minmax_series(out["team_off_rating"]) * 0.10
        + pd.to_numeric(out["team_playoffs"], errors="coerce").fillna(0) * 5.0
    ).clip(0, 100)
    missing_team = out["team_srs"].isna() & out["team_net_rating"].isna()
    out["team_strength_score"] = team_strength
    out.loc[missing_team, "team_strength_score"] = 50.0
    out["team_context_missing"] = missing_team.astype(int)

    # Opponent strength proxy: schedule strength is the cleanest available opponent signal.
    # We also include opponent scoring difficulty through team defensive context as a weak proxy.
    # Future series-level playoff opponents can overwrite/add to these columns.
    opp_proxy_raw = (
        minmax_series(out["team_sos"]) * 0.45
        + minmax_series(out["team_def_rating"] * -1) * 0.15
        + minmax_series(out["team_srs"]) * 0.12
        + minmax_series(out["team_net_rating"]) * 0.08
        + minmax_series(out["playoff_path_difficulty"]) * 0.20
    ).clip(0, 100)
    out["opponent_strength_score"] = opp_proxy_raw
    # If a future playoff_series table exists, it carries more direct opponent information than SOS.
    has_series_context = out["playoff_path_difficulty"].notna()
    out.loc[has_series_context, "opponent_strength_score"] = (
        out.loc[has_series_context, "opponent_strength_score"] * 0.55
        + out.loc[has_series_context, "playoff_path_difficulty"].fillna(50) * 0.45
    ).clip(0, 100)
    out.loc[out["team_sos"].isna() & ~has_series_context, "opponent_strength_score"] = 50.0
    out["opponent_context_missing"] = (out["team_sos"].isna() & ~has_series_context).astype(int)
    out["playoff_series_context_missing"] = (~has_series_context).astype(int)

    # Ability environment correction: hard opponents and weak teammates add credit;
    # strong teammates/strong team context modestly reduce individual credit.
    teammate_centered = 50.0 - out["teammate_strength_score"].fillna(50)
    opponent_centered = out["opponent_strength_score"].fillna(50) - 50.0
    team_centered = 50.0 - out["team_strength_score"].fillna(50)
    out["environment_adjustment"] = (
        teammate_centered * 0.080
        + opponent_centered * 0.070
        + team_centered * 0.035
    ).clip(-8, 8)
    out["environment_context_score"] = (50 + out["environment_adjustment"] * 5.0).clip(0, 100)

    return out


def _add_championship_context(df: pd.DataFrame) -> pd.DataFrame:
    """Classify championships by role and single-core / multi-core context.

    Single-core champion definition used here follows the project rule:
    the champion has no other All-NBA player on the same team-season, and the player himself
    must be an All-NBA player. The leader/core/member tiers are then used to produce a
    championship score. This prevents all championship rings from being treated equally.
    """
    out = df.copy()
    if "playoff_champion" not in out.columns:
        out["playoff_champion"] = 0
    out["playoff_champion"] = pd.to_numeric(out["playoff_champion"], errors="coerce").fillna(0).clip(0, 1)
    for c in ["season_ability_value_base", "playoff_mp", "playoff_ws", "mp", "all_nba_any_flag", "all_nba_score"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["all_nba_any_flag"] = ((out["all_nba_any_flag"] > 0) | (out["all_nba_score"] > 0)).astype(int)

    valid_team = ~out.get("team", pd.Series("", index=out.index)).astype(str).str.upper().isin(TRADED_TEAM_MARKERS)
    # Ranks inside team-season. Smaller rank is better.
    out["team_ability_rank"] = np.nan
    out["team_playoff_mp_rank"] = np.nan
    out["team_playoff_ws_rank"] = np.nan
    if valid_team.any():
        out.loc[valid_team, "team_ability_rank"] = out.loc[valid_team].groupby(["season", "team"])["season_ability_value_base"].rank(method="first", ascending=False)
        out.loc[valid_team, "team_playoff_mp_rank"] = out.loc[valid_team].groupby(["season", "team"])["playoff_mp"].rank(method="first", ascending=False)
        out.loc[valid_team, "team_playoff_ws_rank"] = out.loc[valid_team].groupby(["season", "team"])["playoff_ws"].rank(method="first", ascending=False)

    out["champion_all_nba_count"] = 0.0
    if valid_team.any():
        allnba = out.loc[valid_team].groupby(["season", "team"])["all_nba_any_flag"].transform("sum")
        out.loc[valid_team, "champion_all_nba_count"] = allnba

    champ = out["playoff_champion"].fillna(0).astype(int).eq(1)
    player_allnba = out["all_nba_any_flag"].eq(1)
    out["single_core_champion_flag"] = (champ & player_allnba & ((out["champion_all_nba_count"] - out["all_nba_any_flag"]) <= 0)).astype(int)
    out["championship_leader_flag"] = (champ & ((out["team_ability_rank"] == 1) | (out["team_playoff_ws_rank"] == 1) | (out["team_playoff_mp_rank"] == 1))).astype(int)
    out["championship_core_flag"] = (champ & ((out["team_ability_rank"] <= 3) | (out["team_playoff_ws_rank"] <= 3) | (out["team_playoff_mp_rank"] <= 3))).astype(int)
    out["championship_rotation_flag"] = (champ & ~out["championship_core_flag"].astype(bool) & (out["mp"].fillna(0) >= 500)).astype(int)

    # Ring value by context. Single-core leader is deliberately close to MVP-level value.
    out["championship_score_season"] = 0.0
    single_leader = out["single_core_champion_flag"].eq(1) & out["championship_leader_flag"].eq(1)
    single_other = champ & out["single_core_champion_flag"].eq(0) & (out["champion_all_nba_count"] <= 1) & ~single_leader
    multi_leader = champ & (out["champion_all_nba_count"] >= 2) & out["championship_leader_flag"].eq(1)
    multi_core = champ & (out["champion_all_nba_count"] >= 2) & out["championship_core_flag"].eq(1) & ~multi_leader
    multi_other = champ & (out["champion_all_nba_count"] >= 2) & ~out["championship_core_flag"].astype(bool)

    out.loc[single_leader, "championship_score_season"] = 34.0
    out.loc[single_other & out["championship_core_flag"].eq(1), "championship_score_season"] = 13.0
    out.loc[single_other & ~out["championship_core_flag"].astype(bool), "championship_score_season"] = 5.0
    out.loc[multi_leader, "championship_score_season"] = 25.0
    out.loc[multi_core, "championship_score_season"] = 18.0
    out.loc[multi_other, "championship_score_season"] = 4.5
    out["championship_score_season"] = out["championship_score_season"].clip(0, 36)
    return out


def build_season_features(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> pd.DataFrame:
    regular = tables["regular"].copy()
    playoffs = tables["playoffs"].copy()
    awards = tables["awards"].copy()

    key = ["player_id", "season"]
    df = regular.merge(playoffs, on=key, how="left")
    df = df.merge(awards, on=key, how="left")

    for col in ["mvp_share", "fmvp_share", "dpoy_share", "mvp_award_score", "fmvp_award_score", "dpoy_award_score", "award_vote_score", "all_team_score", "all_nba_score", "all_defense_score", "all_nba_first_score", "all_nba_second_score", "all_nba_third_score", "all_star_score", "award_score_season", "all_nba_any_flag"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for c in ["playoff_bpm", "playoff_ws", "playoff_vorp", "playoff_ts_percent", "playoff_per"]:
        if c not in df.columns:
            df[c] = np.nan
    df["playoff_lift_bpm"] = pd.to_numeric(df["playoff_bpm"], errors="coerce") - pd.to_numeric(df.get("bpm"), errors="coerce")
    df["playoff_lift_ts"] = pd.to_numeric(df["playoff_ts_percent"], errors="coerce") - pd.to_numeric(df.get("ts_percent"), errors="coerce")

    era_cols = [
        "pts_per_100_poss", "ast_per_100_poss", "trb_per_100_poss", "stl_per_100_poss", "blk_per_100_poss",
        "ts_percent", "e_fg_percent", "per", "ws", "ws_48", "ows", "dws", "bpm", "obpm", "dbpm", "vorp", "usg_percent",
        "o_rtg", "d_rtg", "games_pct", "mp"
    ]
    df = zscore_by_group(df, "season", [c for c in era_cols if c in df.columns])

    for col in [c for c in ["pts_per_100_poss", "ast_per_100_poss", "trb_per_100_poss", "bpm", "obpm", "dbpm", "vorp", "ws", "per"] if c in df.columns]:
        x = pd.to_numeric(df[col], errors="coerce")
        grp = df["season"].astype(str) + "_" + df["position_group"].astype(str)
        mu = x.groupby(grp).transform("mean")
        sd = x.groupby(grp).transform("std").replace(0, np.nan)
        df[col + "_pos_z"] = ((x - mu) / sd).clip(-4, 4).fillna(0.0)

    ability_weights = {
        "bpm_era_z": 0.18,
        "vorp_era_z": 0.14,
        "ws_era_z": 0.12,
        "per_era_z": 0.10,
        "obpm_era_z": 0.10,
        "dbpm_era_z": 0.10,
        "pts_per_100_poss_era_z": 0.08,
        "ast_per_100_poss_era_z": 0.05,
        "trb_per_100_poss_era_z": 0.04,
        "ts_percent_era_z": 0.05,
        "games_pct_era_z": 0.04,
    }
    df["season_ability_value_rule"] = robust_score_from_z(df, ability_weights)

    playoff_bonus = (
        minmax_series(df["playoff_bpm"]) * 0.35
        + minmax_series(df["playoff_ws"]) * 0.25
        + minmax_series(df["playoff_vorp"]) * 0.20
        + minmax_series(df["playoff_lift_bpm"]) * 0.10
        + minmax_series(df.get("playoff_g", pd.Series(index=df.index))) * 0.10
    )
    has_playoff = pd.to_numeric(df.get("playoff_g", 0), errors="coerce").fillna(0) > 0
    df["playoff_score_season"] = np.where(has_playoff, playoff_bonus, 40.0)

    df["season_ability_value_base"] = (df["season_ability_value_rule"] * 0.82 + df["playoff_score_season"] * 0.18).clip(0, 100)

    df["season_legacy_value_base"] = (
        df["season_ability_value_base"] * 0.30
        + minmax_series(df["mvp_award_score"].fillna(0) + df["fmvp_award_score"].fillna(0) + df["dpoy_award_score"].fillna(0)) * 0.23
        + minmax_series(df["all_nba_first_score"].fillna(0) + df["all_nba_second_score"].fillna(0) + df["all_nba_third_score"].fillna(0)) * 0.18
        + df["playoff_score_season"] * 0.07
        + minmax_series(df["all_star_score"]) * 0.04
    ).clip(0, 100)

    # Add teammate/opponent/team context after base scores exist, then classify championship value.
    df = _add_team_and_environment_context(df, tables)
    df = _add_championship_context(df)
    df["season_ability_value"] = (
        df["season_ability_value_base"] * 0.86
        + df["playoff_score_season"] * 0.05
        + df["championship_score_season"].fillna(0) * 0.10
        + df["environment_adjustment"].fillna(0)
    ).clip(0, 100)
    df["season_legacy_value"] = (
        df["season_legacy_value_base"] * 0.72
        + df["championship_score_season"].fillna(0) * 0.30
        + df["environment_adjustment"].fillna(0) * 0.35
    ).clip(0, 100)

    df["season_offense_score"] = robust_score_from_z(df, {
        "obpm_era_z": 0.25,
        "pts_per_100_poss_era_z": 0.20,
        "ts_percent_era_z": 0.20,
        "ast_per_100_poss_era_z": 0.15,
        "o_rtg_era_z": 0.10,
        "usg_percent_era_z": 0.10,
    })
    if "d_rtg_era_z" in df.columns:
        df["neg_d_rtg_era_z"] = -df["d_rtg_era_z"]
    df["season_defense_score"] = robust_score_from_z(df, {
        "dbpm_era_z": 0.30,
        "dws_era_z": 0.20,
        "stl_per_100_poss_era_z": 0.10,
        "blk_per_100_poss_era_z": 0.15,
        "trb_per_100_poss_era_z": 0.10,
        "neg_d_rtg_era_z": 0.15,
    })

    feature_dir = Path(out_dir) / "features"
    ensure_dirs(feature_dir)
    df.to_csv(feature_dir / "player_season_features.csv", index=False)
    return df



def build_career_features(season_df: pd.DataFrame, tables: dict[str, pd.DataFrame], out_dir: str | Path) -> pd.DataFrame:
    """Build one-row-per-player career features, including environment context.

    This version is vectorized because the full historical dataset is large enough that
    Python groupby lambdas can become slow on ordinary laptops.
    """
    df = season_df.copy()
    career = tables["career"].copy()
    reputation = tables["reputation"].copy()

    needed = ["season", "g", "mp", "pts", "trb", "ast", "ws", "vorp", "bpm", "season_ability_value", "season_legacy_value",
              "season_offense_score", "season_defense_score", "playoff_score_season", "award_score_season", "mvp_share", "fmvp_share", "dpoy_share", "mvp_award_score", "fmvp_award_score", "dpoy_award_score", "all_nba_score", "all_nba_first_score", "all_nba_second_score", "all_nba_third_score", "all_defense_score", "all_star_score",
              "championship_score_season", "playoff_champion", "single_core_champion_flag", "championship_leader_flag", "championship_core_flag", "championship_rotation_flag", "champion_all_nba_count",
              "playoff_g", "playoff_ws", "playoff_vorp", "teammate_strength_score", "opponent_strength_score", "team_strength_score", "environment_adjustment", "environment_context_score",
              "teammate_ws_sum", "teammate_vorp_sum", "teammate_bpm_weighted_avg", "teammate_award_score_sum", "team_srs", "team_net_rating", "team_win_pct", "team_sos", "playoff_path_difficulty", "playoff_opponent_strength_score", "playoff_opponent_srs_avg", "playoff_opponent_net_rating_avg", "playoff_round_score"]
    for c in needed:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "player" not in df.columns:
        df["player"] = df["player_id"]
    if "player_clean" not in df.columns:
        df["player_clean"] = df["player"].astype(str).str.lower()
    if "position_group" not in df.columns:
        df["position_group"] = "UNK"

    mp = df["mp"].fillna(0)
    df["ability_value_minutes"] = df["season_ability_value"].fillna(50) * np.sqrt(mp.clip(lower=0) / 2400.0).clip(0, 1.25)
    df["legacy_value_minutes"] = df["season_legacy_value"].fillna(50) * np.sqrt(mp.clip(lower=0) / 2400.0).clip(0, 1.25)
    df["quality_flag"] = (df["season_ability_value"].fillna(0) >= 65).astype(int)
    df["elite_flag"] = (df["season_ability_value"].fillna(0) >= 78).astype(int)
    df["bpm_x_mp"] = df["bpm"].fillna(0) * df["mp"].fillna(0).clip(lower=1)
    df["env_x_mp"] = df["environment_adjustment"].fillna(0) * df["mp"].fillna(0).clip(lower=1)

    # Career-stage weighting: rookie/early and twilight seasons matter less as penalties,
    # but excellent early/late seasons create explicit bonus features. Prime years carry
    # the heaviest ability weight. This matches basketball evaluation logic: a player
    # can be forgiven for ordinary rookie/twilight years, while great rookie/old seasons
    # are extra evidence of greatness.
    df = df.sort_values(["player_id", "season"]).copy()
    df["season_index"] = df.groupby("player_id").cumcount() + 1
    df["age_num"] = pd.to_numeric(df.get("age"), errors="coerce")
    career_min = df.groupby("player_id")["season"].transform("min")
    career_max = df.groupby("player_id")["season"].transform("max")
    span = (career_max - career_min).replace(0, np.nan)
    df["career_progress"] = ((df["season"] - career_min) / span).clip(0, 1).fillna(0.5)
    conditions = [
        (df["season_index"] <= 2) | (df["age_num"] <= 23),
        (df["age_num"].between(24, 25, inclusive="both")) | (df["career_progress"] < 0.28),
        (df["age_num"].between(26, 31, inclusive="both")) | (df["career_progress"].between(0.28, 0.62, inclusive="both")),
        (df["age_num"].between(32, 34, inclusive="both")) | (df["career_progress"].between(0.62, 0.82, inclusive="both")),
    ]
    choices = ["rookie_early", "growth", "prime", "late_prime"]
    df["career_phase"] = np.select(conditions, choices, default="twilight")
    phase_weight_map = {"rookie_early": 0.68, "growth": 0.88, "prime": 1.22, "late_prime": 1.00, "twilight": 0.72}
    df["career_phase_weight"] = df["career_phase"].map(phase_weight_map).astype(float)
    df["ability_value_phase_weighted"] = df["season_ability_value"].fillna(50) * np.sqrt(mp.clip(lower=0) / 2400.0).clip(0, 1.25) * df["career_phase_weight"]
    df["legacy_value_phase_weighted"] = df["season_legacy_value"].fillna(50) * np.sqrt(mp.clip(lower=0) / 2400.0).clip(0, 1.25) * (df["career_phase_weight"] * 0.75 + 0.25)

    # Identity fields: keep latest display name, but choose primary position by career minutes.
    df = df.sort_values(["player_id", "season"])
    ident = df.groupby("player_id", sort=False).tail(1)[["player_id", "player", "player_clean"]]
    pos_minutes = df.groupby(["player_id", "position_group"], sort=False)["mp"].sum().reset_index()
    pos_primary = pos_minutes.sort_values(["player_id", "mp"], ascending=[True, False]).drop_duplicates("player_id")[["player_id", "position_group"]]
    ident = ident.merge(pos_primary, on="player_id", how="left")

    numeric_agg = df.groupby("player_id", sort=False).agg(
        career_start=("season", "min"),
        career_end=("season", "max"),
        seasons=("season", "nunique"),
        career_games=("g", "sum"),
        career_minutes=("mp", "sum"),
        career_pts=("pts", "sum"),
        career_trb=("trb", "sum"),
        career_ast=("ast", "sum"),
        career_ws=("ws", "sum"),
        career_vorp=("vorp", "sum"),
        bpm_x_mp=("bpm_x_mp", "sum"),
        env_x_mp=("env_x_mp", "sum"),
        mp_for_bpm=("mp", "sum"),
        career_total_value_raw=("ability_value_minutes", "sum"),
        legacy_total_value_raw=("legacy_value_minutes", "sum"),
        career_total_value_phase_raw=("ability_value_phase_weighted", "sum"),
        legacy_total_value_phase_raw=("legacy_value_phase_weighted", "sum"),
        quality_seasons=("quality_flag", "sum"),
        elite_seasons=("elite_flag", "sum"),
        prime_season_count=("career_phase", lambda s: (s == "prime").sum()),
        rookie_early_season_count=("career_phase", lambda s: (s == "rookie_early").sum()),
        twilight_season_count=("career_phase", lambda s: (s == "twilight").sum()),
        award_score_raw=("award_score_season", "sum"),
        mvp_share_career=("mvp_share", "sum"),
        fmvp_share_career=("fmvp_share", "sum"),
        dpoy_share_career=("dpoy_share", "sum"),
        mvp_award_score_career=("mvp_award_score", "sum"),
        fmvp_award_score_career=("fmvp_award_score", "sum"),
        dpoy_award_score_career=("dpoy_award_score", "sum"),
        all_nba_score_career=("all_nba_score", "sum"),
        all_nba_first_score_career=("all_nba_first_score", "sum"),
        all_nba_second_score_career=("all_nba_second_score", "sum"),
        all_nba_third_score_career=("all_nba_third_score", "sum"),
        all_defense_score_career=("all_defense_score", "sum"),
        all_star_score_career=("all_star_score", "sum"),
        championship_score_raw=("championship_score_season", "sum"),
        championships=("playoff_champion", "sum"),
        single_core_championships=("single_core_champion_flag", "sum"),
        championship_leader_seasons=("championship_leader_flag", "sum"),
        championship_core_seasons=("championship_core_flag", "sum"),
        championship_rotation_seasons=("championship_rotation_flag", "sum"),
        playoff_games_career=("playoff_g", "sum"),
        playoff_ws_career=("playoff_ws", "sum"),
        playoff_vorp_career=("playoff_vorp", "sum"),
        avg_teammate_strength=("teammate_strength_score", "mean"),
        avg_opponent_strength=("opponent_strength_score", "mean"),
        avg_team_strength=("team_strength_score", "mean"),
        avg_environment_context_score=("environment_context_score", "mean"),
        teammate_ws_sum_career_avg=("teammate_ws_sum", "mean"),
        teammate_vorp_sum_career_avg=("teammate_vorp_sum", "mean"),
        teammate_bpm_weighted_career_avg=("teammate_bpm_weighted_avg", "mean"),
        teammate_award_score_career_avg=("teammate_award_score_sum", "mean"),
        avg_team_srs=("team_srs", "mean"),
        avg_team_net_rating=("team_net_rating", "mean"),
        avg_team_win_pct=("team_win_pct", "mean"),
        avg_schedule_strength=("team_sos", "mean"),
        avg_playoff_path_difficulty=("playoff_path_difficulty", "mean"),
        avg_playoff_opponent_strength=("playoff_opponent_strength_score", "mean"),
        avg_playoff_opponent_srs=("playoff_opponent_srs_avg", "mean"),
        max_playoff_round_score=("playoff_round_score", "max"),
    ).reset_index()
    cf = ident.merge(numeric_agg, on="player_id", how="right")
    cf["avg_bpm_weighted"] = cf["bpm_x_mp"] / cf["mp_for_bpm"].replace(0, np.nan)
    cf["environment_adjustment_weighted"] = cf["env_x_mp"] / cf["mp_for_bpm"].replace(0, np.nan)
    cf = cf.drop(columns=["bpm_x_mp", "env_x_mp", "mp_for_bpm"], errors="ignore")
    games = pd.to_numeric(cf["career_games"], errors="coerce").replace(0, np.nan)
    cf["career_pts_per_game"] = pd.to_numeric(cf["career_pts"], errors="coerce") / games
    cf["career_trb_per_game"] = pd.to_numeric(cf["career_trb"], errors="coerce") / games
    cf["career_ast_per_game"] = pd.to_numeric(cf["career_ast"], errors="coerce") / games

    def fast_top_mean(src: str, n: int, name: str) -> pd.DataFrame:
        tmp = df[["player_id", src]].copy()
        tmp[src] = pd.to_numeric(tmp[src], errors="coerce")
        tmp = tmp.dropna(subset=[src]).sort_values(["player_id", src], ascending=[True, False])
        vals = tmp.groupby("player_id", sort=False).head(n).groupby("player_id", sort=False)[src].mean().rename(name).reset_index()
        return vals

    for new_col, src, n in [
        ("peak_1_score", "season_ability_value", 1),
        ("peak_3_score", "season_ability_value", 3),
        ("peak_5_score", "season_ability_value", 5),
        ("legacy_peak_3_score", "season_legacy_value", 3),
        ("offense_score_raw", "season_offense_score", 5),
        ("defense_score_raw", "season_defense_score", 5),
        ("playoff_score_raw", "playoff_score_season", 5),
        ("top5_environment_adjustment", "environment_adjustment", 5),
        ("top5_opponent_strength", "opponent_strength_score", 5),
        ("top5_teammate_strength", "teammate_strength_score", 5),
    ]:
        cf = cf.merge(fast_top_mean(src, n, new_col), on="player_id", how="left")

    def phase_top_mean(phase: str, out_col: str, n: int = 2) -> pd.DataFrame:
        tmp = df.loc[df["career_phase"] == phase, ["player_id", "season_ability_value"]].copy()
        if tmp.empty:
            return pd.DataFrame(columns=["player_id", out_col])
        tmp = tmp.sort_values(["player_id", "season_ability_value"], ascending=[True, False])
        return tmp.groupby("player_id", sort=False).head(n).groupby("player_id", sort=False)["season_ability_value"].mean().rename(out_col).reset_index()

    for phase, col, n in [
        ("rookie_early", "rookie_early_top2_score", 2),
        ("prime", "prime_top5_score", 5),
        ("late_prime", "late_prime_top3_score", 3),
        ("twilight", "twilight_top2_score", 2),
    ]:
        cf = cf.merge(phase_top_mean(phase, col, n), on="player_id", how="left")

    if not career.empty:
        cf = cf.merge(career.drop(columns=["player"], errors="ignore"), on="player_id", how="left")
        cf["hof"] = cf.get("hof", False).fillna(False).astype(bool).astype(int)
    else:
        cf["hof"] = 0
    if "career_start" not in cf.columns and "career_start_x" in cf.columns:
        cf["career_start"] = cf["career_start_x"]
    if "career_end" not in cf.columns and "career_end_x" in cf.columns:
        cf["career_end"] = cf["career_end_x"]
    if "career_start_y" in cf.columns:
        cf["official_career_start"] = cf["career_start_y"]
    if "career_end_y" in cf.columns:
        cf["official_career_end"] = cf["career_end_y"]

    if not reputation.empty:
        cf = cf.merge(reputation.drop_duplicates("player_clean"), on="player_clean", how="left")
    for c in ["wiki_total_views", "importance_proxy"]:
        if c not in cf.columns:
            cf[c] = 0.0
        cf[c] = pd.to_numeric(cf[c], errors="coerce").fillna(0.0)

    cf["career_total_value"] = minmax_series(cf["career_total_value_raw"])
    cf["legacy_total_value"] = minmax_series(cf["legacy_total_value_raw"])
    cf["career_total_value_phase"] = minmax_series(cf["career_total_value_phase_raw"])
    cf["legacy_total_value_phase"] = minmax_series(cf["legacy_total_value_phase_raw"])
    cf["prime_score"] = cf["prime_top5_score"].fillna(cf["peak_5_score"]).clip(0, 100)
    cf["rookie_early_bonus"] = ((cf["rookie_early_top2_score"].fillna(50) - 68).clip(lower=0) / 32 * 5.0).clip(0, 5)
    cf["twilight_bonus"] = ((cf["twilight_top2_score"].fillna(50) - 66).clip(lower=0) / 34 * 5.0).clip(0, 5)
    cf["career_stage_bonus"] = (cf["rookie_early_bonus"] + cf["twilight_bonus"]).clip(0, 8)
    # Longevity matters, but it should not dominate players who merely accumulated totals.
    cf["longevity_score"] = minmax_series(cf["quality_seasons"] * 0.9 + cf["elite_seasons"] * 1.25 + np.sqrt(pd.to_numeric(cf["career_minutes"], errors="coerce").fillna(0).clip(lower=0)) * 0.028)
    cf["career_per_game_score"] = (
        minmax_series(cf["career_pts_per_game"]) * 0.46
        + minmax_series(cf["career_ast_per_game"]) * 0.28
        + minmax_series(cf["career_trb_per_game"]) * 0.18
        + minmax_series(cf["avg_bpm_weighted"]) * 0.08
    ).clip(0, 100)
    cf["offense_score"] = cf["offense_score_raw"].clip(0, 100).fillna(50)
    cf["defense_score"] = cf["defense_score_raw"].clip(0, 100).fillna(50)
    # Non-title playoff rounds count, but are deliberately discounted relative to championship value.
    cf["playoff_score"] = (minmax_series(cf["playoff_score_raw"]) * 0.42 + minmax_series(cf["playoff_ws_career"]) * 0.23 + minmax_series(cf["playoff_vorp_career"]) * 0.15 + minmax_series(cf["max_playoff_round_score"]) * 0.20).clip(0,100)
    cf["championship_score"] = (minmax_series(cf["championship_score_raw"]) * 0.70 + minmax_series(cf["single_core_championships"] * 1.5 + cf["championship_leader_seasons"] * 1.1 + cf["championship_core_seasons"] * 0.8 + cf["championships"] * 0.4) * 0.30).clip(0,100)
    cf["award_score"] = (
        minmax_series(cf["mvp_award_score_career"]) * 0.26
        + minmax_series(cf["fmvp_award_score_career"]) * 0.22
        + minmax_series(cf["all_nba_first_score_career"]) * 0.18
        + minmax_series(cf["dpoy_award_score_career"]) * 0.12
        + minmax_series(cf["all_nba_second_score_career"]) * 0.09
        + minmax_series(cf["all_nba_third_score_career"]) * 0.05
        + minmax_series(cf["all_defense_score_career"]) * 0.04
        + minmax_series(cf["all_star_score_career"]) * 0.04
    ).clip(0,100)
    cf["media_reputation_score"] = (minmax_series(np.log1p(cf["wiki_total_views"])) * 0.65 + minmax_series(np.log1p(cf["importance_proxy"])) * 0.35).clip(0,100)
    cf["environment_difficulty_score"] = (
        minmax_series(50 - cf["avg_teammate_strength"].fillna(50)) * 0.32
        + minmax_series(cf["avg_opponent_strength"].fillna(50)) * 0.22
        + minmax_series(cf["avg_playoff_path_difficulty"].fillna(50)) * 0.18
        + minmax_series(50 - cf["avg_team_strength"].fillna(50)) * 0.16
        + minmax_series(cf["environment_adjustment_weighted"].fillna(0)) * 0.12
    ).clip(0, 100)

    cf["ability_score_rule"] = (
        cf["peak_1_score"].fillna(50) * 0.17
        + cf["peak_3_score"].fillna(50) * 0.18
        + cf["prime_score"].fillna(50) * 0.13
        + cf["peak_5_score"].fillna(50) * 0.08
        + cf["career_per_game_score"] * 0.10
        + cf["championship_score"] * 0.13
        + cf["playoff_score"] * 0.06
        + cf["offense_score"] * 0.06
        + cf["defense_score"] * 0.06
        + cf["environment_difficulty_score"] * 0.05
        + cf["career_total_value_phase"] * 0.04
        + cf["longevity_score"] * 0.02
        + cf["career_stage_bonus"]
    ).clip(0, 100)
    cf["legacy_score_rule"] = (
        cf["championship_score"] * 0.29
        + cf["award_score"] * 0.25
        + cf["legacy_peak_3_score"].fillna(50) * 0.10
        + cf["career_per_game_score"] * 0.08
        + cf["legacy_total_value_phase"] * 0.07
        + cf["playoff_score"] * 0.05
        + cf["longevity_score"] * 0.05
        + cf["media_reputation_score"] * 0.05
        + cf["environment_difficulty_score"] * 0.03
        + cf["career_stage_bonus"] * 0.35
        + cf["hof"] * 5.0
    ).clip(0, 100)
    cf["ability_target"] = (
        cf["ability_score_rule"] * 0.66
        + cf["championship_score"] * 0.12
        + cf["career_per_game_score"] * 0.08
        + minmax_series(cf["avg_bpm_weighted"]) * 0.06
        + cf["environment_difficulty_score"] * 0.04
        + cf["career_total_value_phase"] * 0.03
        + cf["career_stage_bonus"] * 0.01
    ).clip(0,100)
    cf["legacy_target"] = (
        cf["legacy_score_rule"] * 0.62
        + cf["championship_score"] * 0.17
        + cf["award_score"] * 0.12
        + cf["media_reputation_score"] * 0.04
        + cf["environment_difficulty_score"] * 0.03
        + cf["hof"] * 5.0
        + cf["career_stage_bonus"] * 0.01
    ).clip(0,100)

    feature_dir = Path(out_dir) / "features"
    ensure_dirs(feature_dir)
    cf.to_csv(feature_dir / "player_career_features.csv", index=False)
    return cf
