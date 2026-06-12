from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import (
    choose_one_row_per_player_season,
    ensure_dirs,
    find_base_dir,
    normalize_name,
    primary_position,
    read_csv,
    season_to_int,
    minmax_series,
)


def _prep_player_season_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "season" in df.columns:
        df["season"] = df["season"].map(season_to_int).astype("Int64")
    if "lg" in df.columns:
        df = df[df["lg"].astype(str).str.upper().isin(["NBA", "BAA", "ABA", "NAN"])]
    if "player" in df.columns:
        df["player_clean"] = df["player"].map(normalize_name)
    if "pos" in df.columns:
        df["position_group"] = df["pos"].map(primary_position)
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(str)
    return df


def load_clean_regular(base_dir: str | Path) -> pd.DataFrame:
    base = find_base_dir(base_dir)
    src = base / "nba_aba_baa_stats"

    totals = _prep_player_season_df(read_csv(src / "Player Totals.csv"))
    per_game = _prep_player_season_df(read_csv(src / "Player Per Game.csv"))
    per100 = _prep_player_season_df(read_csv(src / "Per 100 Poss.csv"))
    adv = _prep_player_season_df(read_csv(src / "Advanced.csv"))
    p36 = _prep_player_season_df(read_csv(src / "Per 36 Minutes.csv"))

    totals = choose_one_row_per_player_season(totals)
    per_game = choose_one_row_per_player_season(per_game)
    per100 = choose_one_row_per_player_season(per100)
    adv = choose_one_row_per_player_season(adv)
    p36 = choose_one_row_per_player_season(p36)

    key = ["player_id", "season"]
    keep_id_cols = ["player_id", "season", "player", "age", "team", "pos", "position_group", "g", "gs", "mp"]
    regular = totals[[c for c in keep_id_cols + [
        "fg", "fga", "x3p", "x3pa", "x2p", "x2pa", "ft", "fta", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "pts", "trp_dbl",
        "fg_percent", "x3p_percent", "e_fg_percent", "ft_percent"
    ] if c in totals.columns]].copy()

    per_game_keep = ["mp_per_game", "pts_per_game", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game"]
    regular = regular.merge(per_game[key + [c for c in per_game_keep if c in per_game.columns]], on=key, how="left")

    per100_keep = [c for c in per100.columns if c.endswith("_per_100_poss") or c in ["o_rtg", "d_rtg"]]
    regular = regular.merge(per100[key + per100_keep], on=key, how="left")

    p36_keep = [c for c in p36.columns if c.endswith("_per_36_min")]
    regular = regular.merge(p36[key + p36_keep], on=key, how="left")

    adv_keep = ["per", "ts_percent", "x3p_ar", "f_tr", "orb_percent", "drb_percent", "trb_percent", "ast_percent", "stl_percent", "blk_percent", "tov_percent", "usg_percent", "ows", "dws", "ws", "ws_48", "obpm", "dbpm", "bpm", "vorp"]
    regular = regular.merge(adv[key + [c for c in adv_keep if c in adv.columns]], on=key, how="left")

    regular["player_clean"] = regular["player"].map(normalize_name)
    regular["games_pct"] = pd.to_numeric(regular.get("g"), errors="coerce") / 82.0
    regular["games_pct"] = regular["games_pct"].clip(0, 1.1)

    # Rule-era availability indicators; not zero-filling old missing values.
    regular["has_3pt_line"] = (regular["season"] >= 1980).astype(int)
    regular["has_steal_block_stat"] = (regular["season"] >= 1974).astype(int)
    regular["has_turnover_stat"] = (regular["season"] >= 1978).astype(int)
    for col in ["x3p", "x3pa", "stl", "blk", "tov", "x3p_per_100_poss", "stl_per_100_poss", "blk_per_100_poss", "tov_per_100_poss"]:
        if col in regular.columns:
            regular[f"missing_{col}"] = regular[col].isna().astype(int)

    return regular


def load_clean_playoffs(base_dir: str | Path, regular: pd.DataFrame) -> pd.DataFrame:
    base = find_base_dir(base_dir)
    f = base / "nba_playoff_players" / "playoffStats.csv"
    if not f.exists():
        return pd.DataFrame(columns=["player_id", "season"])
    po = read_csv(f)
    po.columns = [c.strip() for c in po.columns]
    po["season"] = po["season"].map(season_to_int).astype("Int64")
    po["player_clean"] = po["player"].map(normalize_name)

    # Map playoff dataset by clean name + season to player_id from regular data.
    mapper = regular[["player_id", "player_clean", "season"]].drop_duplicates()
    po = po.merge(mapper, on=["player_clean", "season"], how="left")

    rename = {
        "team_id": "playoff_team",
        "g": "playoff_g",
        "gs": "playoff_gs",
        "mp": "playoff_mp",
        "mp_per_g": "playoff_mp_per_game",
        "pts_per_g": "playoff_pts_per_game",
        "trb_per_g": "playoff_trb_per_game",
        "ast_per_g": "playoff_ast_per_game",
        "stl_per_g": "playoff_stl_per_game",
        "blk_per_g": "playoff_blk_per_game",
        "tov_per_g": "playoff_tov_per_game",
        "ts_pct": "playoff_ts_percent",
        "usg_pct": "playoff_usg_percent",
        "bpm": "playoff_bpm",
        "obpm": "playoff_obpm",
        "dbpm": "playoff_dbpm",
        "ws": "playoff_ws",
        "ws_per_48": "playoff_ws_48",
        "vorp": "playoff_vorp",
        "per": "playoff_per",
    }
    po = po.rename(columns=rename)
    cols = ["player_id", "player", "player_clean", "season", "playoff_team"] + [c for c in po.columns if c.startswith("playoff_")]
    po = po[[c for c in cols if c in po.columns]]
    # Some traded playoff rows are rare; aggregate safely.
    numeric_cols = [c for c in po.columns if c.startswith("playoff_") and c not in ["playoff_team"]]
    agg = {c: "sum" if c in ["playoff_g", "playoff_gs", "playoff_mp", "playoff_ws", "playoff_vorp"] else "mean" for c in numeric_cols}
    keep = po.groupby(["player_id", "season"], dropna=False).agg(agg).reset_index()
    return keep


def load_awards(base_dir: str | Path) -> pd.DataFrame:
    """Load and tier awards with basketball-specific weights.

    Key logic used by the ranking model:
    MVP >= FMVP >= All-NBA 1st >= DPOY > All-NBA 2nd > All-NBA 3rd / All-Defense.
    Award shares are used when available; All-NBA/All-Defense/All-Star rows are treated
    as binary season achievements.
    """
    base = find_base_dir(base_dir)
    src = base / "nba_aba_baa_stats"
    rows = []

    shares_f = src / "Player Award Shares.csv"
    if shares_f.exists():
        shares = read_csv(shares_f)
        shares["season"] = shares["season"].map(season_to_int).astype("Int64")
        shares["award_norm"] = shares["award"].astype(str).str.lower()
        shares["share"] = pd.to_numeric(shares.get("share"), errors="coerce").fillna(0)
        for _, r in shares.iterrows():
            award = r["award_norm"]
            share = float(r["share"])
            rec = {
                "player_id": str(r["player_id"]),
                "season": r["season"],
                "mvp_share": 0.0,
                "fmvp_share": 0.0,
                "dpoy_share": 0.0,
                "mvp_award_score": 0.0,
                "fmvp_award_score": 0.0,
                "dpoy_award_score": 0.0,
                "other_award_score": 0.0,
            }
            if "finals" in award and "mvp" in award:
                rec["fmvp_share"] = share
                rec["fmvp_award_score"] = 30.0 * share
            elif "mvp" in award:
                rec["mvp_share"] = share
                rec["mvp_award_score"] = 35.0 * share
            elif "dpoy" in award or "defensive player" in award:
                rec["dpoy_share"] = share
                rec["dpoy_award_score"] = 20.0 * share
            elif "roy" in award:
                rec["other_award_score"] = 3.0 * share
            elif "sixth" in award or "smoy" in award:
                rec["other_award_score"] = 3.0 * share
            elif "mip" in award or "clutch" in award:
                rec["other_award_score"] = 2.0 * share
            rows.append(rec)

    eos_f = src / "End of Season Teams.csv"
    if eos_f.exists():
        eos = read_csv(eos_f)
        eos["season"] = eos["season"].map(season_to_int).astype("Int64")
        for _, r in eos.iterrows():
            typ = str(r.get("type", "")).lower()
            num = str(r.get("number_tm", "")).lower()
            rec = {
                "player_id": str(r["player_id"]),
                "season": r["season"],
                "all_team_score": 0.0,
                "all_nba_score": 0.0,
                "all_defense_score": 0.0,
                "all_nba_first_score": 0.0,
                "all_nba_second_score": 0.0,
                "all_nba_third_score": 0.0,
                "all_defense_first_score": 0.0,
                "all_defense_second_score": 0.0,
                "all_nba_first_flag": 0,
                "all_nba_second_flag": 0,
                "all_nba_third_flag": 0,
                "all_nba_any_flag": 0,
            }
            if "all-nba" in typ:
                if "1st" in num:
                    rec["all_nba_first_score"] = 24.0
                    rec["all_nba_first_flag"] = 1
                elif "2nd" in num:
                    rec["all_nba_second_score"] = 13.0
                    rec["all_nba_second_flag"] = 1
                elif "3rd" in num:
                    rec["all_nba_third_score"] = 7.0
                    rec["all_nba_third_flag"] = 1
                else:
                    rec["all_nba_third_score"] = 5.0
                rec["all_nba_score"] = rec["all_nba_first_score"] + rec["all_nba_second_score"] + rec["all_nba_third_score"]
                rec["all_team_score"] = rec["all_nba_score"]
                rec["all_nba_any_flag"] = 1
            elif "all-defense" in typ:
                if "1st" in num:
                    rec["all_defense_first_score"] = 6.0
                elif "2nd" in num:
                    rec["all_defense_second_score"] = 3.5
                else:
                    rec["all_defense_second_score"] = 2.0
                rec["all_defense_score"] = rec["all_defense_first_score"] + rec["all_defense_second_score"]
                rec["all_team_score"] = rec["all_defense_score"]
            rows.append(rec)

    as_f = src / "All-Star Selections.csv"
    if as_f.exists():
        ast = read_csv(as_f)
        ast["season"] = ast["season"].map(season_to_int).astype("Int64")
        for _, r in ast.iterrows():
            rows.append({"player_id": str(r["player_id"]), "season": r["season"], "all_star_score": 2.0, "all_star_flag": 1})

    if not rows:
        return pd.DataFrame(columns=["player_id", "season"])
    awards = pd.DataFrame(rows)
    stable = [
        "mvp_share", "fmvp_share", "dpoy_share", "mvp_award_score", "fmvp_award_score", "dpoy_award_score", "other_award_score",
        "all_team_score", "all_nba_score", "all_defense_score", "all_nba_first_score", "all_nba_second_score", "all_nba_third_score",
        "all_defense_first_score", "all_defense_second_score", "all_star_score", "all_star_flag",
        "all_nba_first_flag", "all_nba_second_flag", "all_nba_third_flag", "all_nba_any_flag",
    ]
    for col in stable:
        if col not in awards.columns:
            awards[col] = 0.0
    awards = awards.groupby(["player_id", "season"], as_index=False).sum(numeric_only=True)
    # Cap binary flags after aggregation.
    for c in ["all_star_flag", "all_nba_first_flag", "all_nba_second_flag", "all_nba_third_flag", "all_nba_any_flag"]:
        awards[c] = awards[c].clip(0, 1)
    awards["award_vote_score"] = awards[["mvp_award_score", "fmvp_award_score", "dpoy_award_score", "other_award_score"]].sum(axis=1)
    awards["award_score_season"] = awards[["award_vote_score", "all_team_score", "all_star_score"]].sum(axis=1)
    return awards


def load_career_info(base_dir: str | Path) -> pd.DataFrame:
    base = find_base_dir(base_dir)
    f = base / "nba_aba_baa_stats" / "Player Career Info.csv"
    info = read_csv(f)
    info["player_id"] = info["player_id"].astype(str)
    info["career_start"] = pd.to_numeric(info.get("from"), errors="coerce")
    info["career_end"] = pd.to_numeric(info.get("to"), errors="coerce")
    info["position_group_career"] = info.get("pos", pd.Series(index=info.index, dtype="object")).map(primary_position)
    return info[[c for c in ["player_id", "player", "pos", "position_group_career", "ht_in_in", "wt", "birth_date", "career_start", "career_end", "hof"] if c in info.columns]]


def load_reputation(base_dir: str | Path) -> pd.DataFrame:
    base = find_base_dir(base_dir)
    rep_dir = base / "reputation_data" / "processed"
    dfs = []
    f1 = rep_dir / "wiki_current_total_views.csv"
    if f1.exists():
        d = read_csv(f1)
        d["player_clean"] = d["player_name"].map(normalize_name)
        dfs.append(d[["player_clean", "wiki_total_views"]])
    f2 = rep_dir / "reputation_player_list.csv"
    if f2.exists():
        d = read_csv(f2)
        d["player_clean"] = d["player_name"].map(normalize_name)
        dfs.append(d[["player_clean", "importance_proxy"]])
    if not dfs:
        return pd.DataFrame(columns=["player_clean"])
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="player_clean", how="outer")
    return out




def _build_team_name_abbrev_map(base_dir: str | Path) -> pd.DataFrame:
    base = find_base_dir(base_dir)
    f = base / "nba_aba_baa_stats" / "Team Abbrev.csv"
    if not f.exists():
        return pd.DataFrame(columns=["season", "team_full_name", "team"])
    m = read_csv(f)
    m.columns = [c.strip() for c in m.columns]
    m["season"] = m["season"].map(season_to_int).astype("Int64")
    m["team_full_name"] = m.get("team", pd.Series(index=m.index, dtype="object")).astype(str).str.strip()
    m["team_full_clean"] = m["team_full_name"].map(normalize_name)
    m["team"] = m.get("abbreviation", pd.Series(index=m.index, dtype="object")).astype(str).str.strip()
    return m[["season", "team_full_name", "team_full_clean", "team"]].drop_duplicates()


# Historical aliases that sometimes appear differently in Basketball-Reference tables.
_TEAM_ALIAS_TO_CURRENT_BREF = {
    "charlotte bobcats": "Charlotte Hornets",
    "new orleans hornets": "New Orleans Pelicans",
    "new jersey nets": "Brooklyn Nets",
    "seattle supersonics": "Oklahoma City Thunder",
    "washington bullets": "Washington Wizards",
    "capital bullets": "Washington Wizards",
    "baltimore bullets": "Washington Wizards",
    "san francisco warriors": "Golden State Warriors",
    "philadelphia warriors": "Golden State Warriors",
    "syracuse nationals": "Philadelphia 76ers",
    "minneapolis lakers": "Los Angeles Lakers",
    "fort wayne pistons": "Detroit Pistons",
    "rochester royals": "Sacramento Kings",
    "cincinnati royals": "Sacramento Kings",
    "kansas city kings": "Sacramento Kings",
    "kansas city-omaha kings": "Sacramento Kings",
    "st. louis hawks": "Atlanta Hawks",
    "tri-cities blackhawks": "Atlanta Hawks",
    "buffalo braves": "Los Angeles Clippers",
    "san diego clippers": "Los Angeles Clippers",
    "vancouver grizzlies": "Memphis Grizzlies",
    "new orleans/oklahoma city hornets": "New Orleans Pelicans",
}


def _map_series_team_to_abbrev(name: object, season: object, mapper: pd.DataFrame) -> str:
    text = str(name).strip()
    if not text or text.lower() == "nan":
        return ""
    # Remove seed: "Chicago Bulls (1)" -> "Chicago Bulls"
    import re
    text_no_seed = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    clean = normalize_name(text_no_seed)
    season_int = season_to_int(season)
    if not mapper.empty:
        sub = mapper[(mapper["season"] == season_int) & (mapper["team_full_clean"] == clean)]
        if not sub.empty:
            return str(sub.iloc[0]["team"])
        # Fallback: some names are current franchise aliases; use same season if possible.
        alias = _TEAM_ALIAS_TO_CURRENT_BREF.get(clean)
        if alias:
            sub = mapper[(mapper["season"] == season_int) & (mapper["team_full_clean"] == normalize_name(alias))]
            if not sub.empty:
                return str(sub.iloc[0]["team"])
        # Last resort across seasons: handles relocation names present only in old seasons.
        sub = mapper[mapper["team_full_clean"] == clean]
        if not sub.empty:
            # Prefer closest season row.
            tmp = sub.copy()
            tmp["dist"] = (pd.to_numeric(tmp["season"], errors="coerce") - season_int).abs()
            return str(tmp.sort_values("dist").iloc[0]["team"])
    return text_no_seed


def load_team_context(base_dir: str | Path) -> pd.DataFrame:
    """Load team-season context and compute a preliminary team power score.

    The preliminary team_power_score is intentionally computed only from team-season
    information, so it can safely be used as an opponent-strength input before the
    richer roster-based team ranking is built later.
    """
    base = find_base_dir(base_dir)
    f = base / "nba_aba_baa_stats" / "Team Summaries.csv"
    if not f.exists():
        return pd.DataFrame(columns=["season", "team"])
    tm = read_csv(f)
    tm.columns = [c.strip() for c in tm.columns]
    if "lg" in tm.columns:
        tm = tm[tm["lg"].astype(str).str.upper().isin(["NBA", "BAA", "ABA", "NAN"])]
    tm["season"] = tm["season"].map(season_to_int).astype("Int64")
    tm["team_name"] = tm.get("team", "").astype(str)
    tm["team"] = tm.get("abbreviation", tm.get("team", "")).astype(str)
    tm["team_win_pct"] = pd.to_numeric(tm.get("w"), errors="coerce") / (pd.to_numeric(tm.get("w"), errors="coerce") + pd.to_numeric(tm.get("l"), errors="coerce")).replace(0, np.nan)
    rename = {
        "srs": "team_srs",
        "sos": "team_sos",
        "o_rtg": "team_off_rating",
        "d_rtg": "team_def_rating",
        "n_rtg": "team_net_rating",
        "pace": "team_pace",
        "mov": "team_mov",
        "w": "team_wins",
        "l": "team_losses",
        "playoffs": "team_playoffs",
        "opp_e_fg_percent": "team_opp_efg_percent",
        "opp_tov_percent": "team_opp_tov_percent",
        "drb_percent": "team_drb_percent",
        "opp_ft_fga": "team_opp_ft_fga",
    }
    tm = tm.rename(columns=rename)
    keep = ["season", "team", "team_name", "lg", "team_wins", "team_losses", "team_win_pct", "team_srs", "team_sos", "team_off_rating", "team_def_rating", "team_net_rating", "team_pace", "team_mov", "team_playoffs", "team_opp_efg_percent", "team_opp_tov_percent", "team_drb_percent", "team_opp_ft_fga"]
    out = tm[[c for c in keep if c in tm.columns]].drop_duplicates(["season", "team"])
    if "team_playoffs" in out.columns:
        out["team_playoffs"] = out["team_playoffs"].astype(str).str.lower().isin(["true", "1", "yes", "y"]).astype(int)
    for c in ["team_wins", "team_losses", "team_win_pct", "team_srs", "team_sos", "team_off_rating", "team_def_rating", "team_net_rating", "team_pace", "team_mov"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # League-year z scores for era-relative dominance.
    for c in ["team_srs", "team_net_rating", "team_win_pct", "team_mov", "team_off_rating"]:
        x = out[c]
        mu = x.groupby(out["season"]).transform("mean")
        sd = x.groupby(out["season"]).transform("std").replace(0, np.nan)
        out[c + "_era_z"] = ((x - mu) / sd).clip(-4, 4).fillna(0)
    # Lower defensive rating is better.
    x = -out["team_def_rating"]
    out["team_def_rating_inv_era_z"] = ((x - x.groupby(out["season"]).transform("mean")) / x.groupby(out["season"]).transform("std").replace(0, np.nan)).clip(-4, 4).fillna(0)
    out["team_power_score_prelim"] = (
        (out["team_srs_era_z"] * 0.34 + out["team_net_rating_era_z"] * 0.24 + out["team_win_pct_era_z"] * 0.18 + out["team_off_rating_era_z"] * 0.10 + out["team_def_rating_inv_era_z"] * 0.10 + out["team_playoffs"].fillna(0) * 0.04) * 12 + 50
    ).clip(0, 100)
    return out


def load_playoff_series(base_dir: str | Path, team_context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load playoff series table and summarize opponent path difficulty by team-season.

    Supports files produced by Basketball-Reference "Table as CSV", the earlier
    cleaned file, or already-normalized opponent rows. Team full names are mapped to
    Basketball-Reference abbreviations so this table can connect to player/team rows.
    """
    base = find_base_dir(base_dir)
    candidates = [
        base / "playoff_series.csv",
        base / "playoff_series_clean.csv",
        base / "nba_playoff_series" / "playoff_series.csv",
        base / "nba_playoff_series" / "playoff_series_clean.csv",
        base / "raw" / "playoff_series.csv",
        base / "raw" / "playoff_series_clean.csv",
    ]
    f = next((x for x in candidates if x.exists()), None)
    stable_cols = [
        "season", "team", "playoff_series_count", "playoff_series_wins", "playoff_series_losses",
        "playoff_round_score", "playoff_path_difficulty", "playoff_opponent_srs_avg",
        "playoff_opponent_net_rating_avg", "playoff_opponent_win_pct_avg", "playoff_opponent_power_avg", "playoff_opponent_strength_score",
        "playoff_champion", "playoff_finals", "playoff_upset_wins",
    ]
    if f is None:
        return pd.DataFrame(columns=stable_cols)

    raw = read_csv(f)
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    # Remove duplicate unnamed columns from BRef CSV export.
    raw = raw[[c for c in raw.columns if not c.startswith("unnamed")]].copy()
    if "season" not in raw.columns and "yr" in raw.columns:
        raw["season"] = raw["yr"]
    raw["season"] = raw["season"].map(season_to_int).astype("Int64")

    mapper = _build_team_name_abbrev_map(base_dir)

    def round_score(x):
        t = str(x).strip().lower()
        if "tiebreak" in t:
            return 0
        if "first round" in t or "quarterfinal" in t:
            return 1
        # Check semifinals before generic finals, because "semifinals" contains "finals".
        if "semifinals" in t or "semi-finals" in t or "semis" in t:
            return 2
        if "conf finals" in t or "conference finals" in t or "div finals" in t or "division finals" in t:
            return 3
        if t == "finals" or t.endswith(" finals"):
            return 4
        return np.nan

    rows = []
    # Common BRef Table as CSV / cleaned columns: winner, loser, winner_wins, loser_wins.
    if {"winner", "loser"}.issubset(raw.columns):
        for _, r in raw.iterrows():
            season = r.get("season")
            rnd = r.get("series", r.get("round", ""))
            rw = round_score(rnd)
            ww = pd.to_numeric(r.get("winner_wins", r.get("w", np.nan)), errors="coerce")
            lw = pd.to_numeric(r.get("loser_wins", r.get("w.1", np.nan)), errors="coerce")
            win_name = str(r.get("winner", "")).strip()
            lose_name = str(r.get("loser", "")).strip()
            # Skip in-progress series lines that Table as CSV sometimes contains.
            if not win_name or not lose_name or "lead the" in win_name.lower() or "lead the" in lose_name.lower():
                continue
            win = _map_series_team_to_abbrev(win_name, season, mapper)
            lose = _map_series_team_to_abbrev(lose_name, season, mapper)
            favorite = str(r.get("favorite", ""))
            underdog = str(r.get("underdog", ""))
            if win:
                rows.append({"season": season, "team": win, "opponent_team": lose, "series_round": rnd, "round_score": rw, "series_result": "W", "team_wins": ww, "opponent_wins": lw, "team_full_name": win_name, "opponent_full_name": lose_name, "favorite": favorite, "underdog": underdog})
            if lose:
                rows.append({"season": season, "team": lose, "opponent_team": win, "series_round": rnd, "round_score": rw, "series_result": "L", "team_wins": lw, "opponent_wins": ww, "team_full_name": lose_name, "opponent_full_name": win_name, "favorite": favorite, "underdog": underdog})
    elif {"winner_team", "loser_team"}.issubset(raw.columns):
        for _, r in raw.iterrows():
            season = r.get("season")
            rnd = r.get("series_round", r.get("round", r.get("series", "")))
            rw = round_score(rnd)
            ww = pd.to_numeric(r.get("winner_wins", r.get("w_wins", r.get("winner_w", np.nan))), errors="coerce")
            lw = pd.to_numeric(r.get("loser_wins", r.get("l_wins", r.get("loser_w", np.nan))), errors="coerce")
            win = str(r.get("winner_team", "")).strip()
            lose = str(r.get("loser_team", "")).strip()
            if win:
                rows.append({"season": season, "team": win, "opponent_team": lose, "series_round": rnd, "round_score": rw, "series_result": "W", "team_wins": ww, "opponent_wins": lw})
            if lose:
                rows.append({"season": season, "team": lose, "opponent_team": win, "series_round": rnd, "round_score": rw, "series_result": "L", "team_wins": lw, "opponent_wins": ww})
    elif {"team", "opponent_team"}.issubset(raw.columns):
        for _, r in raw.iterrows():
            rnd = r.get("series_round", r.get("round", ""))
            rows.append({
                "season": r.get("season"), "team": str(r.get("team", "")).strip(),
                "opponent_team": str(r.get("opponent_team", "")).strip(),
                "series_round": rnd, "round_score": round_score(rnd),
                "series_result": r.get("result", r.get("series_result", "")),
                "team_wins": pd.to_numeric(r.get("team_wins", np.nan), errors="coerce"),
                "opponent_wins": pd.to_numeric(r.get("opponent_wins", np.nan), errors="coerce"),
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(columns=stable_cols)

    if team_context is not None and not team_context.empty:
        opp = team_context.rename(columns={
            "team": "opponent_team",
            "team_srs": "opponent_srs",
            "team_net_rating": "opponent_net_rating",
            "team_win_pct": "opponent_win_pct",
            "team_power_score_prelim": "opponent_power_score_prelim",
        })
        detail = detail.merge(opp[[c for c in ["season", "opponent_team", "opponent_srs", "opponent_net_rating", "opponent_win_pct", "opponent_power_score_prelim"] if c in opp.columns]], on=["season", "opponent_team"], how="left")
    for c in ["opponent_srs", "opponent_net_rating", "opponent_win_pct", "opponent_power_score_prelim", "round_score"]:
        if c not in detail.columns:
            detail[c] = np.nan
        detail[c] = pd.to_numeric(detail[c], errors="coerce")
    detail["series_win_flag"] = detail["series_result"].astype(str).str.upper().str.startswith("W").astype(int)
    detail["series_loss_flag"] = detail["series_result"].astype(str).str.upper().str.startswith("L").astype(int)
    detail["finals_flag"] = detail["round_score"].eq(4).astype(int)
    detail["champion_flag"] = ((detail["round_score"] == 4) & (detail["series_win_flag"] == 1)).astype(int)
    detail["upset_win_flag"] = 0
    # If odds text is available, flag simple underdog wins: winner abbreviation appears in underdog field.
    if "underdog" in detail.columns:
        detail.loc[(detail["series_win_flag"] == 1) & detail.apply(lambda r: str(r.get("team", "")) in str(r.get("underdog", "")), axis=1), "upset_win_flag"] = 1
    detail["opponent_strength_raw"] = (
        detail["opponent_power_score_prelim"].fillna(50) * 0.55
        + (detail["opponent_srs"].fillna(0) * 5 + 50).clip(0, 100) * 0.20
        + (detail["opponent_net_rating"].fillna(0) * 5 + 50).clip(0, 100) * 0.15
        + detail["opponent_win_pct"].fillna(0.5) * 100 * 0.10
    )
    detail["series_difficulty_raw"] = detail["opponent_strength_raw"] * (1 + detail["round_score"].fillna(1) * 0.14)
    agg = detail.groupby(["season", "team"], dropna=False).agg(
        playoff_series_count=("opponent_team", "count"),
        playoff_series_wins=("series_win_flag", "sum"),
        playoff_series_losses=("series_loss_flag", "sum"),
        playoff_round_score=("round_score", "max"),
        playoff_opponent_srs_avg=("opponent_srs", "mean"),
        playoff_opponent_net_rating_avg=("opponent_net_rating", "mean"),
        playoff_opponent_win_pct_avg=("opponent_win_pct", "mean"),
        playoff_opponent_power_avg=("opponent_power_score_prelim", "mean"),
        playoff_path_difficulty_raw=("series_difficulty_raw", "mean"),
        playoff_champion=("champion_flag", "max"),
        playoff_finals=("finals_flag", "max"),
        playoff_upset_wins=("upset_win_flag", "sum"),
    ).reset_index()
    agg["playoff_opponent_strength_score"] = minmax_series(agg["playoff_path_difficulty_raw"])
    agg["playoff_path_difficulty"] = (
        agg["playoff_opponent_strength_score"].fillna(50) * 0.62
        + minmax_series(agg["playoff_round_score"].fillna(0)) * 0.18
        + minmax_series(agg["playoff_series_wins"].fillna(0)) * 0.12
        + agg["playoff_champion"].fillna(0) * 8
    ).clip(0, 100)
    return agg[stable_cols]

def build_clean_tables(base_dir: str | Path, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    clean_dir = out_dir / "cleaned"
    ensure_dirs(clean_dir)
    regular = load_clean_regular(base_dir)
    playoffs = load_clean_playoffs(base_dir, regular)
    awards = load_awards(base_dir)
    career = load_career_info(base_dir)
    reputation = load_reputation(base_dir)
    team_context = load_team_context(base_dir)
    playoff_series = load_playoff_series(base_dir, team_context)

    regular.to_csv(clean_dir / "regular_clean.csv", index=False)
    playoffs.to_csv(clean_dir / "playoffs_clean.csv", index=False)
    awards.to_csv(clean_dir / "awards_clean.csv", index=False)
    career.to_csv(clean_dir / "players_clean.csv", index=False)
    reputation.to_csv(clean_dir / "reputation_clean.csv", index=False)
    team_context.to_csv(clean_dir / "team_context_clean.csv", index=False)
    playoff_series.to_csv(clean_dir / "playoff_series_context_clean.csv", index=False)
    return {"regular": regular, "playoffs": playoffs, "awards": awards, "career": career, "reputation": reputation, "team_context": team_context, "playoff_series": playoff_series}
