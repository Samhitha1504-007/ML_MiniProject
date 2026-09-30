import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "postseason_prediction_data.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. SELECT FEATURES
# --------------------------------------------------

features = [
    "home_pts_scored",
    "home_pts_allowed",
    "home_pass_yards",
    "home_rush_yards",
    "home_total_yards",
    "home_turnovers",
    "home_comp_pct",
    "home_pen_yards",
    "home_win",
    "home_point_diff",

    "away_pts_scored",
    "away_pts_allowed",
    "away_pass_yards",
    "away_rush_yards",
    "away_total_yards",
    "away_turnovers",
    "away_comp_pct",
    "away_pen_yards",
    "away_win",
    "away_point_diff"
]


X = df[features]


# Actual postseason scores
y_home = df["score_home"]
y_away = df["score_away"]


# Actual score difference
y_difference = df["score_difference"]


# --------------------------------------------------
# 3. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

# Use earlier seasons for training
# and later seasons for testing.

train = df[df["season"] < 2023].copy()
test = df[df["season"] >= 2023].copy()

print("Training games:", len(train))
print("Testing games:", len(test))

print(
    "Training seasons:",
    train["season"].min(),
    "to",
    train["season"].max()
)

print(
    "Testing seasons:",
    test["season"].min(),
    "to",
    test["season"].max()
)


X_train = train[features]
X_test = test[features]


# --------------------------------------------------
# 4. SCORE REGRESSION
# --------------------------------------------------

home_score_model = LinearRegression()

away_score_model = LinearRegression()


# Train separate models for home and away scores
home_score_model.fit(
    X_train,
    train["score_home"]
)

away_score_model.fit(
    X_train,
    train["score_away"]
)


# Predict scores
pred_home_score = home_score_model.predict(X_test)

pred_away_score = away_score_model.predict(X_test)


# --------------------------------------------------
# 5. SCORE PREDICTION EVALUATION
# --------------------------------------------------

home_mae = mean_absolute_error(
    test["score_home"],
    pred_home_score
)

away_mae = mean_absolute_error(
    test["score_away"],
    pred_away_score
)


home_rmse = np.sqrt(
    mean_squared_error(
        test["score_home"],
        pred_home_score
    )
)

away_rmse = np.sqrt(
    mean_squared_error(
        test["score_away"],
        pred_away_score
    )
)


print("\n--- SCORE REGRESSION ---")

print(
    "Home Score MAE:",
    round(home_mae, 2)
)

print(
    "Away Score MAE:",
    round(away_mae, 2)
)

print(
    "Home Score RMSE:",
    round(home_rmse, 2)
)

print(
    "Away Score RMSE:",
    round(away_rmse, 2)
)


# --------------------------------------------------
# 6. PREDICT WINNER USING PREDICTED SCORES
# --------------------------------------------------

predicted_winner = (
    pred_home_score > pred_away_score
).astype(int)

actual_winner = test["winner"]


winner_accuracy = accuracy_score(
    actual_winner,
    predicted_winner
)


print(
    "\nWinner Accuracy:",
    round(winner_accuracy * 100, 2),
    "%"
)


# --------------------------------------------------
# 7. DIFFERENTIAL REGRESSION
# --------------------------------------------------

difference_model = LinearRegression()

difference_model.fit(
    X_train,
    train["score_difference"]
)


pred_difference = difference_model.predict(
    X_test
)


# Predict winner from score difference
difference_winner = (
    pred_difference > 0
).astype(int)


difference_accuracy = accuracy_score(
    actual_winner,
    difference_winner
)


difference_mae = mean_absolute_error(
    test["score_difference"],
    pred_difference
)


difference_rmse = np.sqrt(
    mean_squared_error(
        test["score_difference"],
        pred_difference
    )
)


print("\n--- DIFFERENTIAL REGRESSION ---")

print(
    "Difference MAE:",
    round(difference_mae, 2)
)

print(
    "Difference RMSE:",
    round(difference_rmse, 2)
)

print(
    "Winner Accuracy:",
    round(difference_accuracy * 100, 2),
    "%"
)


# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    actual_winner,
    difference_winner
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Winner")
plt.ylabel("Actual Winner")

plt.xticks(
    [0, 1],
    ["Away Win", "Home Win"]
)

plt.yticks(
    [0, 1],
    ["Away Win", "Home Win"]
)

plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=150
)

plt.show()


# --------------------------------------------------
# 9. SAVE PREDICTIONS
# --------------------------------------------------

results = test[
    [
        "season",
        "date",
        "away",
        "home",
        "score_away",
        "score_home"
    ]
].copy()

results["actual_winner"] = np.where(
    test["winner"] == 1,
    results["home"],
    results["away"]
)

results["predicted_away_score"] = (
    pred_away_score.round(2)
)

results["predicted_home_score"] = (
    pred_home_score.round(2)
)

results["predicted_difference"] = (
    pred_difference.round(2)
)

results["predicted_winner"] = np.where(
    difference_winner == 1,
    results["home"],
    results["away"]
)

results.to_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "model_predictions.csv"
    ),
    index=False
)


print("\nPredictions saved successfully.")

print("\nSample predictions:")

print(
    results.head(10).to_string(index=False)
)