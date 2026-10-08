"""
Why this project no longer uses the Pima Indians Diabetes Dataset.

The first version of CuraDiet trained on Pima (768 rows) and topped out
around 74-77% accuracy. This script shows that's a property of the data,
not of the model: three quite different classifiers, evaluated with
honest 5-fold cross-validation (imputation fitted inside each fold), all
plateau in the same place.

Run: python -m src.benchmark_pima
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PIMA_DATA_PATH, RANDOM_STATE

# 0 means "missing" in these columns (nobody has 0 blood pressure)
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def main():
    df = pd.read_csv(PIMA_DATA_PATH)
    df[ZERO_AS_MISSING_COLS] = df[ZERO_AS_MISSING_COLS].replace(0, np.nan)
    X, y = df.drop(columns="Outcome"), df["Outcome"]

    models = {
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=RANDOM_STATE),
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=RANDOM_STATE
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"Pima: {len(df)} rows, majority-class baseline {1 - y.mean():.3f}\n")
    for name, model in models.items():
        r = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
        print(
            f"  {name:22s} accuracy {r['test_accuracy'].mean():.3f}"
            f"   ROC-AUC {r['test_roc_auc'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
