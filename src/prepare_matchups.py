import pandas as pd
import os


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_FILE = os.path.join(
    DATA_DIR,
    "cfb_box-scores_2002-2025.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "postseason_prediction_data.csv"
)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(RAW_FILE, low_memory=False)

print("Original dataset:", df.shape)


# --------------------------------------------------
# 2. SEPARATE REGULAR AND POSTSEASON GAMES
# --------------------------------------------------

regular = df[df["game_type"] == "regular"].copy()

postseason = df[df["game_type"] == "post"].copy()

print("Regular-season games:", len(regular))
print("Postseason games:", len(postseason))


# --------------------------------------------------
# 3. CREATE TEAM GAME RECORDS
# --------------------------------------------------

records = []

for _, row in regular.iterrows():

    # Home team
    records.append({
        "season": row["season"],
        "team": row["home"],

        "pts_scored": row["score_home"],
        "pts_allowed": row["score_away"],

        "pass_yards": row["pass_yards_home"],
        "rush_yards": row["rush_yards_home"],
        "total_yards": row["total_yards_home"],

        "turnovers": row["fum_home"] + row["int_home"],

        "comp_pct":
            row["pass_comp_home"] / row["pass_att_home"]
            if row["pass_att_home"] != 0 else None,

        "pen_yards": row["pen_yards_home"]
    })


    # Away team
    records.append({
        "season": row["season"],
        "team": row["away"],

        "pts_scored": row["score_away"],
        "pts_allowed": row["score_home"],

        "pass_yards": row["pass_yards_away"],
        "rush_yards": row["rush_yards_away"],
        "total_yards": row["total_yards_away"],

        "turnovers": row["fum_away"] + row["int_away"],

        "comp_pct":
            row["pass_comp_away"] / row["pass_att_away"]
            if row["pass_att_away"] != 0 else None,

        "pen_yards": row["pen_yards_away"]
    })


team_games = pd.DataFrame(records)


# --------------------------------------------------
# 4. CALCULATE REGULAR-SEASON AVERAGES
# --------------------------------------------------

stats = [
    "pts_scored",
    "pts_allowed",
    "pass_yards",
    "rush_yards",
    "total_yards",
    "turnovers",
    "comp_pct",
    "pen_yards"
]

team_averages = (
    team_games
    .groupby(["season", "team"])[stats]
    .mean()
    .reset_index()
)

print("\nTeam-season averages created:")
print(team_averages.head())


# --------------------------------------------------
# 5. ADD HOME TEAM STATISTICS
# --------------------------------------------------

home_stats = team_averages.copy()

home_stats = home_stats.rename(
    columns={
        "team": "home"
    }
)

for stat in stats:
    home_stats = home_stats.rename(
        columns={
            stat: "home_" + stat
        }
    )


# --------------------------------------------------
# 6. ADD AWAY TEAM STATISTICS
# --------------------------------------------------

away_stats = team_averages.copy()

away_stats = away_stats.rename(
    columns={
        "team": "away"
    }
)

for stat in stats:
    away_stats = away_stats.rename(
        columns={
            stat: "away_" + stat
        }
    )


# --------------------------------------------------
# 7. MERGE TEAM STATISTICS WITH POSTSEASON GAMES
# --------------------------------------------------

postseason_data = postseason[
    [
        "season",
        "date",
        "away",
        "home",
        "score_away",
        "score_home"
    ]
].copy()


# Add home-team statistics
postseason_data = postseason_data.merge(
    home_stats,
    on=["season", "home"],
    how="left"
)


# Add away-team statistics
postseason_data = postseason_data.merge(
    away_stats,
    on=["season", "away"],
    how="left"
)


# --------------------------------------------------
# 8. REMOVE GAMES WITHOUT REGULAR-SEASON HISTORY
# --------------------------------------------------

feature_columns = []

for stat in stats:
    feature_columns.append("home_" + stat)
    feature_columns.append("away_" + stat)


before = len(postseason_data)

postseason_data = postseason_data.dropna(
    subset=feature_columns
)

after = len(postseason_data)

print("\nPostseason games removed due to missing team statistics:",
      before - after)


# --------------------------------------------------
# 9. CREATE SCORE DIFFERENCE
# --------------------------------------------------

postseason_data["score_difference"] = (
    postseason_data["score_home"]
    - postseason_data["score_away"]
)


# --------------------------------------------------
# 10. CREATE WINNER
# --------------------------------------------------

postseason_data["winner"] = (
    postseason_data["score_home"]
    > postseason_data["score_away"]
).astype(int)


# --------------------------------------------------
# 11. SAVE DATASET
# --------------------------------------------------

postseason_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFinal dataset:", postseason_data.shape)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFinal columns:")
print(postseason_data.columns.tolist())