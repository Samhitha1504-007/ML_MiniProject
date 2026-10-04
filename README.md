# College Football Game Outcome Predictor

## Overview
This is a machine learning mini-project for UE24CS352A. The goal is to predict the outcome of college football postseason (bowl) games using regular-season team performance statistics. 

Instead of looking at individual team stats in isolation, our model compares the two teams by calculating the difference between their average stats (a "Differential" approach).

## Dataset
We used a dataset containing college football games from 2002 to 2025. 
* **Training Data:** 2002–2022 seasons
* **Testing Data:** 2023–2025 seasons (Out-of-time evaluation)

## Features Used
We engineered 7 "differential" features by subtracting the away team's average stats from the home team's average stats:
- Points Scored Difference
- Points Allowed Difference
- Total Yards Difference
- Turnovers Difference
- Completion Percentage Difference
- Penalty Yards Difference
- Win Percentage Difference

## Machine Learning Model
The primary model used is **Logistic Regression** (with L2 regularization, C=0.1). 

We chose Logistic Regression because it easily handles binary classification (Home Win vs. Away Win) and outputs win probabilities, allowing us to assign a "Confidence Level" (High, Medium, Low) to every prediction.

## Project Structure
```text
ML_MiniProject/
│
├── data/
│   ├── cfb_box-scores_2002-2025.csv       # Raw box score data
│   ├── cfb_cleaned.csv                    # Cleaned historical data
│   ├── postseason_prediction_data.csv     # Processed dataset for model training
│   ├── model_predictions.csv              # Baseline Linear Regression outputs
│   ├── random_forest_predictions.csv      # Baseline Random Forest outputs
│   └── improved_model_predictions.csv     # Final Logistic Regression outputs with confidence scores
│
├── src/
│   ├── preprocessing.py                   # Initial data cleaning pipeline
│   ├── prepare_matchups.py                # Feature engineering (team averages & matchups)
│   ├── model.py                           # Baseline Linear Regression model
│   ├── random_forest_model.py             # Baseline Random Forest model
│   └── improved_model.py                  # Final Logistic Regression model (Differential approach)
│
├── results/
│   └── improved_confusion_matrix.png      # Visualization of the final model performance
│
├── EDA.ipynb                              # Exploratory Data Analysis notebook
├── requirements.txt                       # Project dependencies
└── README.md                              # Project documentation
```

## Installation
1. Clone this repository to your local machine.
2. Open a terminal in the project folder.
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
To run the model and view the evaluation metrics, execute the following command from the root project folder:

**Windows:**
```bash
python src\improved_model.py
```
**Mac/Linux:**
```bash
python src/improved_model.py
```

The script will output the model's accuracy, a classification report, and generate a confusion matrix image in the `results/` folder. It also saves the predicted outcomes and confidence levels to the `data/` folder.

## Evaluation
* **Target:** Predict if the Home Team wins (1) or Away Team wins (0).
* **Accuracy:** ~66% on unseen test data (2023-2025).

## Team Members
* Samhitha S
* Rohil Thimmanahalli Girish