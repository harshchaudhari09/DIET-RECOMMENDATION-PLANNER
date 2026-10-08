"""
Shared settings for the diabetes risk model + diet recommendation app.
Keeping this in one place means the training script and the Flask
app can't drift out of sync on feature names or paths.

All paths are resolved relative to the project root, so the app and
the training script work no matter which directory they're run from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "diabetes_prediction_dataset.csv"
PIMA_DATA_PATH = DATA_DIR / "pima_diabetes.csv"  # legacy dataset, see README

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "diabetes_risk_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42

# Features the diabetes risk model is trained on (Diabetes Prediction
# Dataset, ~100k patient records -- see README for source & limitations)
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "diabetes"

GENDERS = ["Female", "Male"]

# Smoking-history categories exactly as they appear in the dataset,
# mapped to the label shown in the form
SMOKING_OPTIONS = {
    "never": "Never smoked",
    "former": "Former smoker",
    "not current": "Not currently smoking",
    "ever": "Have smoked at some point",
    "current": "Current smoker",
    "No Info": "Prefer not to say",
}

# Activity multipliers for the Mifflin-St Jeor TDEE calculation used
# by the diet recommendation engine (src/diet_rules.py)
ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,       # little/no exercise
    "light": 1.375,         # light exercise 1-3 days/week
    "moderate": 1.55,       # moderate exercise 3-5 days/week
    "active": 1.725,        # hard exercise 6-7 days/week
}

# Accepted input ranges for the web form: (min, max, label). Anything
# outside these is almost certainly a typo or a unit mix-up (e.g. glucose
# in mmol/L instead of mg/dL), so we reject it with a clear message
# instead of letting the model extrapolate.
INPUT_RANGES = {
    "age": (18, 100, "Age"),
    "height_cm": (120, 230, "Height"),
    "weight_kg": (30, 300, "Weight"),
    "glucose": (50, 400, "Blood glucose"),
    "hba1c": (3.5, 15.0, "HbA1c"),
    "systolic": (70, 250, "Systolic blood pressure"),
    "diastolic": (40, 150, "Diastolic blood pressure"),
}
