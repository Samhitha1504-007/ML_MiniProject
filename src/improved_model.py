import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "postseason_prediction_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

df = pd.read_csv(DATA_FILE)
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. FEATURE ENGINEERING (DIFFERENTIAL STATISTICS)
# --------------------------------------------------
df["diff_pts_scored"] = df["home_pts_scored"] - df["away_pts_scored"]
df["diff_pts_allowed"] = df["home_pts_allowed"] - df["away_pts_allowed"]
df["diff_total_yards"] = df["home_total_yards"] - df["away_total_yards"]
df["diff_turnovers"] = df["home_turnovers"] - df["away_turnovers"]
df["diff_comp_pct"] = df["home_comp_pct"] - df["away_comp_pct"]
df["diff_pen_yards"] = df["home_pen_yards"] - df["away_pen_yards"]
df["diff_win_pct"] = df["home_win"] - df["away_win"]

features = [
    "diff_pts_scored", 
    "diff_pts_allowed", 
    "diff_total_yards", 
    "diff_turnovers",
    "diff_comp_pct", 
    "diff_pen_yards",
    "diff_win_pct"
]

# --------------------------------------------------
# 3. TIME-BASED TRAIN / TEST SPLIT
# --------------------------------------------------
train = df[df["season"] < 2023].copy()
test = df[df["season"] >= 2023].copy()

X_train = train[features]
X_test = test[features]
y_train = train["winner"]
y_test = test["winner"]

# --------------------------------------------------
# 4. FEATURE SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 5. LOGISTIC REGRESSION (DIFFERENTIAL MODEL)
# --------------------------------------------------
model = LogisticRegression(C=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict winner and probabilities (for confidence)
predicted_winner = model.predict(X_test_scaled)
predicted_probs = model.predict_proba(X_test_scaled)

# --------------------------------------------------
# 6. EVALUATION & CONFIDENCE
# --------------------------------------------------
winner_accuracy = accuracy_score(y_test, predicted_winner)

print("\n--- IMPROVED DIFFERENTIAL MODEL ---")
print("Winner Accuracy:", round(winner_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predicted_winner))

# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(y_test, predicted_winner)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Improved Model Confusion Matrix")
plt.xlabel("Predicted Winner")
plt.ylabel("Actual Winner")
plt.xticks([0, 1], ["Away Win", "Home Win"])
plt.yticks([0, 1], ["Away Win", "Home Win"])
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="red" if cm[i,j] < 30 else "white")

plt.tight_layout()
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(os.path.join(RESULTS_DIR, "improved_confusion_matrix.png"), dpi=150)
print("\nConfusion matrix saved to results folder.")

# --------------------------------------------------
# 8. SAVE PREDICTIONS WITH CONFIDENCE
# --------------------------------------------------
results = test[["season", "date", "away", "home", "score_away", "score_home"]].copy()

results["actual_winner"] = np.where(test["winner"] == 1, results["home"], results["away"])
results["predicted_winner"] = np.where(predicted_winner == 1, results["home"], results["away"])

results["win_probability"] = np.max(predicted_probs, axis=1)

conditions = [
    (results["win_probability"] >= 0.65),
    (results["win_probability"] >= 0.55) & (results["win_probability"] < 0.65),
    (results["win_probability"] < 0.55)
]
choices = ["High", "Medium", "Low"]
results["confidence_level"] = np.select(conditions, choices, default="Unknown")
results["correct_prediction"] = (results["actual_winner"] == results["predicted_winner"])

results.to_csv(os.path.join(BASE_DIR, "data", "improved_model_predictions.csv"), index=False)
print("\nSample predictions with confidence:")
print(results.head(10).to_string(index=False))
