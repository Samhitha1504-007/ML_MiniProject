import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
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


# --------------------------------------------------
# 3. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------

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
# 4. RANDOM FOREST REGRESSION
# --------------------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# Train model to predict score difference

rf_model.fit(
    X_train,
    train["score_difference"]
)


# Predict score difference

pred_difference = rf_model.predict(
    X_test
)


# --------------------------------------------------
# 5. PREDICT WINNER
# --------------------------------------------------

predicted_winner = (
    pred_difference > 0
).astype(int)

actual_winner = test["winner"]


winner_accuracy = accuracy_score(
    actual_winner,
    predicted_winner
)


print("\n--- RANDOM FOREST ---")

print(
    "Winner Accuracy:",
    round(winner_accuracy * 100, 2),
    "%"
)


# --------------------------------------------------
# 6. DIFFERENCE EVALUATION
# --------------------------------------------------

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


print(
    "Difference MAE:",
    round(difference_mae, 2)
)

print(
    "Difference RMSE:",
    round(difference_rmse, 2)
)


# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    actual_winner,
    predicted_winner
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Random Forest Confusion Matrix")

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
        "random_forest_confusion_matrix.png"
    ),
    dpi=150
)

plt.show()


# --------------------------------------------------
# 8. SAVE PREDICTIONS
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


# Actual winner

results["actual_winner"] = np.where(
    test["winner"] == 1,
    results["home"],
    results["away"]
)


# Predicted score difference

results["predicted_difference"] = (
    pred_difference.round(2)
)


# Predicted winner

results["predicted_winner"] = np.where(
    predicted_winner == 1,
    results["home"],
    results["away"]
)


# Check whether prediction was correct

results["correct_prediction"] = (
    results["actual_winner"] ==
    results["predicted_winner"]
)


results.to_csv(
    os.path.join(
        BASE_DIR,
        "data",
        "random_forest_predictions.csv"
    ),
    index=False
)


print("\nRandom Forest predictions saved successfully.")

print("\nSample predictions:")

print(
    results.head(10).to_string(index=False)
)