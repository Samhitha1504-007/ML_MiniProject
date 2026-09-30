"""
preprocessing.py
Member 1 — Data + Preprocessing
College Football Bowl Game Predictor
UE24CS352A Machine Learning Mini-Project

Dataset columns (cfb_box-scores_2002-2025.csv):
  season, week, date, time_et, game_type,
  away, home, rank_away, rank_home, conf_away, conf_home, neutral,
  score_away, score_home,
  q1/q2/q3/q4/ot (away & home),
  first_downs, third_down, fourth_down, pass_comp, pass_att, pass_yards,
  rush_att, rush_yards, total_yards, fum, int, pen_num, pen_yards,
  possession, attendance, tv  (away & home variants)
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
RAW_CSV   = os.path.join(DATA_DIR, "cfb_box-scores_2002-2025.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "cfb_cleaned.csv")


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"[1] Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"    Raw shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 2. CLEAN & SORT
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("[2] Cleaning...")

    # Parse date, sort chronologically
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop rows without scores or team names
    before = len(df)
    df.dropna(subset=["score_home", "score_away", "home", "away"], inplace=True)
    print(f"    Dropped {before - len(df)} rows with missing scores/teams.")

    # Convert numeric columns
    num_cols = [
        "score_home", "score_away",
        "pass_yards_home", "pass_yards_away",
        "rush_yards_home", "rush_yards_away",
        "total_yards_home", "total_yards_away",
        "fum_home", "fum_away",
        "int_home", "int_away",
        "first_downs_home", "first_downs_away",
        "pen_yards_home", "pen_yards_away",
        "pen_num_home", "pen_num_away",
        "pass_comp_home", "pass_comp_away",
        "pass_att_home", "pass_att_away",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute turnovers = fumbles + interceptions
    if "fum_home" in df.columns and "int_home" in df.columns:
        df["turnovers_home"] = df["fum_home"].fillna(0) + df["int_home"].fillna(0)
        df["turnovers_away"] = df["fum_away"].fillna(0) + df["int_away"].fillna(0)

    # Compute completion percentage
    if "pass_comp_home" in df.columns and "pass_att_home" in df.columns:
        df["comp_pct_home"] = df["pass_comp_home"] / df["pass_att_home"].replace(0, np.nan)
        df["comp_pct_away"] = df["pass_comp_away"] / df["pass_att_away"].replace(0, np.nan)

    print(f"    Shape after cleaning: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. TARGET VARIABLE
# ─────────────────────────────────────────────
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    print("[3] Creating target variable...")
    # 1 = home wins, 0 = away wins (drop ties)
    ties = (df["score_home"] == df["score_away"]).sum()
    df = df[df["score_home"] != df["score_away"]].copy()
    df["target"] = (df["score_home"] > df["score_away"]).astype(int)
    print(f"    Ties dropped: {ties}")
    print(f"    Home wins: {df['target'].sum()} | Away wins: {(df['target']==0).sum()}")
    return df


# ─────────────────────────────────────────────
# 4. ROLLING HISTORICAL FEATURES
# ─────────────────────────────────────────────
def build_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For each game, compute each team's rolling average over its
    last `window` games BEFORE this game (no data leakage).

    Stats per team:
      pts_scored, pts_allowed, pass_yards, rush_yards,
      total_yards, turnovers, comp_pct, pen_yards
    """
    print(f"[4] Building rolling features (window={window})...")

    # ── Build per-team game log ───────────────────────────────────────
    records = []
    for _, row in df.iterrows():
        base = {"date": row["date"]}

        # Home team stats in this game
        records.append({**base,
            "team"       : row["home"],
            "pts_scored" : row["score_home"],
            "pts_allowed": row["score_away"],
            "pass_yards" : row.get("pass_yards_home",  np.nan),
            "rush_yards" : row.get("rush_yards_home",  np.nan),
            "total_yards": row.get("total_yards_home", np.nan),
            "turnovers"  : row.get("turnovers_home",   np.nan),
            "comp_pct"   : row.get("comp_pct_home",    np.nan),
            "pen_yards"  : row.get("pen_yards_home",   np.nan),
        })
        # Away team stats in this game
        records.append({**base,
            "team"       : row["away"],
            "pts_scored" : row["score_away"],
            "pts_allowed": row["score_home"],
            "pass_yards" : row.get("pass_yards_away",  np.nan),
            "rush_yards" : row.get("rush_yards_away",  np.nan),
            "total_yards": row.get("total_yards_away", np.nan),
            "turnovers"  : row.get("turnovers_away",   np.nan),
            "comp_pct"   : row.get("comp_pct_away",    np.nan),
            "pen_yards"  : row.get("pen_yards_away",   np.nan),
        })

    game_log = pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)

    stat_cols = ["pts_scored", "pts_allowed", "pass_yards",
                 "rush_yards", "total_yards", "turnovers", "comp_pct", "pen_yards"]

    # ── Rolling avg (shift 1 to avoid leakage) ───────────────────────
    game_log[stat_cols] = game_log[stat_cols].apply(pd.to_numeric, errors="coerce")

    rolled = (
        game_log.groupby("team")[stat_cols]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    rolled.columns = [f"avg_{c}" for c in stat_cols]
    game_log = pd.concat([game_log, rolled], axis=1)

    # ── Merge home features back ──────────────────────────────────────
    home_feats = (game_log
        .rename(columns={"team": "home", **{f"avg_{c}": f"home_avg_{c}" for c in stat_cols}})
        [["home", "date"] + [f"home_avg_{c}" for c in stat_cols]]
    )
    away_feats = (game_log
        .rename(columns={"team": "away", **{f"avg_{c}": f"away_avg_{c}" for c in stat_cols}})
        [["away", "date"] + [f"away_avg_{c}" for c in stat_cols]]
    )

    df = df.merge(home_feats, on=["home", "date"], how="left")
    df = df.merge(away_feats, on=["away", "date"], how="left")

    feat_cols = [f"home_avg_{c}" for c in stat_cols] + [f"away_avg_{c}" for c in stat_cols]
    before = len(df)
    df.dropna(subset=feat_cols, inplace=True)
    print(f"    Dropped {before - len(df)} rows with insufficient history.")
    print(f"    Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────
def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"[5] Saved cleaned dataset -> {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  College Football Predictor — Preprocessing")
    print("=" * 60)

    df = load_data(RAW_CSV)
    df = clean_data(df)
    df = create_target(df)
    df = build_rolling_features(df, window=5)
    save_data(df, CLEAN_CSV)

    feat_cols = [c for c in df.columns if c.startswith("home_avg_") or c.startswith("away_avg_")]
    print("\n[DONE] Feature columns available for model.py:")
    for c in feat_cols:
        print(f"   - {c}")
    print(f"\n   Total games in cleaned dataset: {len(df)}")
    print(f"   Home win rate: {df['target'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()
