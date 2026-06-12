from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dirs

DROP_COLS = {
    "player_id", "player", "player_clean", "position_group", "position_group_career", "pos", "birth_date",
    "ability_target", "legacy_target", "ability_score_final", "legacy_score_final", "final_score",
}


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in DROP_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool:
            cols.append(c)
    return cols


def _fit_predict_ensemble(X: pd.DataFrame, y: pd.Series, target_name: str, model_dir: Path) -> tuple[np.ndarray, dict]:
    valid = y.notna()
    Xv = X.loc[valid]
    yv = y.loc[valid]
    idx = np.arange(len(Xv))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=18, max_depth=7, min_samples_leaf=6, random_state=42, n_jobs=-1)),
    ])
    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(random_state=42, n_estimators=18, learning_rate=0.06, max_depth=2)),
    ])
    rf.fit(Xv.iloc[train_idx], yv.iloc[train_idx])
    gb.fit(Xv.iloc[train_idx], yv.iloc[train_idx])
    p_test = rf.predict(Xv.iloc[test_idx]) * 0.55 + gb.predict(Xv.iloc[test_idx]) * 0.45
    metrics = {
        "target": target_name,
        "mae": float(mean_absolute_error(yv.iloc[test_idx], p_test)),
        "r2": float(r2_score(yv.iloc[test_idx], p_test)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    joblib.dump({"rf": rf, "gb": gb, "features": list(X.columns)}, model_dir / f"{target_name}_ml_ensemble.joblib")
    return np.clip(rf.predict(X) * 0.55 + gb.predict(X) * 0.45, 0, 100), metrics


def _fit_predict_mlp(X: pd.DataFrame, y: pd.Series, target_name: str, model_dir: Path) -> tuple[np.ndarray, dict]:
    valid = y.notna()
    Xv = X.loc[valid]
    yv = y.loc[valid]
    idx = np.arange(len(Xv))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    mlp = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(36, 18), activation="relu", alpha=1e-4, learning_rate_init=8e-4, max_iter=35, early_stopping=True, n_iter_no_change=5, random_state=42)),
    ])
    mlp.fit(Xv.iloc[train_idx], yv.iloc[train_idx])
    p_test = mlp.predict(Xv.iloc[test_idx])
    metrics = {
        "target": target_name,
        "mae": float(mean_absolute_error(yv.iloc[test_idx], p_test)),
        "r2": float(r2_score(yv.iloc[test_idx], p_test)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
    joblib.dump({"mlp": mlp, "features": list(X.columns)}, model_dir / f"{target_name}_deep_mlp.joblib")
    return np.clip(mlp.predict(X), 0, 100), metrics


def train_and_rank(career_df: pd.DataFrame, out_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    model_dir = out_dir / "models"
    outputs_dir = out_dir / "outputs"
    ensure_dirs(model_dir, outputs_dir)

    df = career_df.copy()
    feature_cols = numeric_feature_cols(df)
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    ability_ml, m1 = _fit_predict_ensemble(X, df["ability_target"], "ability", model_dir)
    legacy_ml, m2 = _fit_predict_ensemble(X, df["legacy_target"], "legacy", model_dir)
    ability_deep, m3 = _fit_predict_mlp(X, df["ability_target"], "ability", model_dir)
    legacy_deep, m4 = _fit_predict_mlp(X, df["legacy_target"], "legacy", model_dir)

    df["ability_score_ml"] = ability_ml
    df["legacy_score_ml"] = legacy_ml
    df["ability_score_deep"] = ability_deep
    df["legacy_score_deep"] = legacy_deep

    df["ability_score_final"] = (df["ability_score_rule"] * 0.60 + df["ability_score_ml"] * 0.25 + df["ability_score_deep"] * 0.15).clip(0, 100)
    df["legacy_score_final"] = (df["legacy_score_rule"] * 0.60 + df["legacy_score_ml"] * 0.25 + df["legacy_score_deep"] * 0.15).clip(0, 100)
    df["final_score"] = (df["ability_score_final"] * 0.55 + df["legacy_score_final"] * 0.45).clip(0, 100)

    df["eligible"] = ((df["career_minutes"].fillna(0) >= 5000) | (df["hof"].fillna(0) == 1) | (df["seasons"].fillna(0) >= 5)).astype(int)
    df = df.sort_values("legacy_score_final", ascending=False).reset_index(drop=True)
    eligible = df["eligible"] == 1
    df.loc[eligible, "legacy_rank"] = df.loc[eligible, "legacy_score_final"].rank(method="first", ascending=False).astype(int)
    df.loc[eligible, "ability_rank"] = df.loc[eligible, "ability_score_final"].rank(method="first", ascending=False).astype(int)
    df.loc[eligible, "final_rank"] = df.loc[eligible, "final_score"].rank(method="first", ascending=False).astype(int)
    df["position_ability_rank"] = np.nan
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        mask = (df["position_group"] == pos) & eligible
        df.loc[mask, "position_ability_rank"] = df.loc[mask, "ability_score_final"].rank(method="first", ascending=False).astype(int)

    keep_cols = [
        "legacy_rank", "ability_rank", "position_ability_rank", "player", "player_id", "position_group", "career_start", "career_end",
        "legacy_score_final", "ability_score_final", "final_score", "peak_1_score", "peak_3_score", "peak_5_score", "prime_score",
        "career_total_value", "career_total_value_phase", "legacy_total_value_phase", "career_per_game_score", "career_pts_per_game", "career_trb_per_game", "career_ast_per_game", "longevity_score", "rookie_early_top2_score", "twilight_top2_score", "career_stage_bonus",
        "offense_score", "defense_score", "playoff_score", "championship_score", "championship_score_raw", "championships", "single_core_championships", "championship_leader_seasons", "championship_core_seasons", "award_score", "mvp_share_career", "fmvp_share_career", "dpoy_share_career", "all_nba_first_score_career", "all_nba_second_score_career", "all_nba_third_score_career", "media_reputation_score",
        "environment_difficulty_score", "avg_teammate_strength", "avg_opponent_strength", "avg_playoff_path_difficulty", "avg_playoff_opponent_srs", "avg_team_strength", "environment_adjustment_weighted",
        "career_games", "career_minutes", "career_pts", "career_trb", "career_ast", "career_ws", "career_vorp", "hof", "eligible"
    ]
    df.to_csv(outputs_dir / "ranking_results.csv", index=False)
    df[keep_cols].sort_values("legacy_score_final", ascending=False).to_csv(outputs_dir / "ranking_legacy_overall.csv", index=False)
    df[keep_cols].sort_values("ability_score_final", ascending=False).to_csv(outputs_dir / "ranking_ability_overall.csv", index=False)
    for pos in ["PG", "SG", "SF", "PF", "C"]:
        df.loc[df["position_group"] == pos, keep_cols].sort_values("ability_score_final", ascending=False).to_csv(outputs_dir / f"ranking_ability_{pos}.csv", index=False)

    metrics = {"ml_ability": m1, "ml_legacy": m2, "deep_mlp_ability": m3, "deep_mlp_legacy": m4, "n_players": int(len(df)), "n_eligible": int(df["eligible"].sum()), "features": feature_cols}
    with open(model_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return df
