"""
Turns a health profile into a diabetes risk assessment.

Two independent signals are combined:

  1. The ML model's probability (src/train_model.py), bucketed into
     Low / Elevated / High using the thresholds saved with the model.
  2. Hard clinical cut-offs from the American Diabetes Association
     (ADA) diagnostic criteria. These don't depend on the model at
     all: if HbA1c or glucose is already in the diabetes range, the
     app says so plainly, whatever the model thinks.
"""

from dataclasses import dataclass, field
from typing import Optional

import joblib
import pandas as pd
import sklearn

from src.config import MODEL_PATH

# ADA "Standards of Care" thresholds
HBA1C_DIABETES = 6.5       # % -- diabetes range
HBA1C_PREDIABETES = 5.7    # % -- prediabetes range (5.7-6.4)
RANDOM_GLUCOSE_DIABETES = 200   # mg/dL, random (non-fasting) glucose
FASTING_GLUCOSE_DIABETES = 126  # mg/dL
FASTING_GLUCOSE_PREDIABETES = 100  # mg/dL


class ModelNotTrainedError(RuntimeError):
    pass


def load_model_bundle(path=MODEL_PATH) -> dict:
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {path}. Run `python -m src.train_model` first."
        )
    bundle = joblib.load(path)
    if bundle.get("sklearn_version") != sklearn.__version__:
        # Pickled sklearn models aren't guaranteed to load correctly across
        # versions -- this was the most likely cause of the old app breaking
        # on other machines. Warn loudly rather than fail silently.
        print(
            f"WARNING: model was trained with scikit-learn {bundle.get('sklearn_version')} "
            f"but {sklearn.__version__} is installed. Re-run `python -m src.train_model`."
        )
    return bundle


@dataclass
class RiskAssessment:
    probability: float
    level: str              # "Low" | "Elevated" | "High"
    clinical_flags: list = field(default_factory=list)
    in_diabetes_range: bool = False
    in_prediabetes_range: bool = False

    @property
    def needs_glycemic_diet(self) -> bool:
        """Should the diet engine switch to a lower-carb, ADA-style plan?"""
        return self.level != "Low" or self.in_diabetes_range or self.in_prediabetes_range


def clinical_flags(hba1c: float, glucose: float, fasting: bool):
    """ADA diagnostic-range checks. Returns (flags, in_diabetes_range, in_prediabetes_range)."""
    flags = []
    diabetes = prediabetes = False

    if hba1c >= HBA1C_DIABETES:
        diabetes = True
        flags.append(
            f"HbA1c of {hba1c:.1f}% is in the diabetes range (≥ {HBA1C_DIABETES}%). "
            "Please confirm with a doctor."
        )
    elif hba1c >= HBA1C_PREDIABETES:
        prediabetes = True
        flags.append(f"HbA1c of {hba1c:.1f}% is in the prediabetes range (5.7–6.4%).")

    if fasting:
        if glucose >= FASTING_GLUCOSE_DIABETES:
            diabetes = True
            flags.append(
                f"Fasting glucose of {glucose:.0f} mg/dL is in the diabetes range "
                f"(≥ {FASTING_GLUCOSE_DIABETES}). Please confirm with a doctor."
            )
        elif glucose >= FASTING_GLUCOSE_PREDIABETES:
            prediabetes = True
            flags.append(
                f"Fasting glucose of {glucose:.0f} mg/dL is in the prediabetes range (100–125)."
            )
    elif glucose >= RANDOM_GLUCOSE_DIABETES:
        diabetes = True
        flags.append(
            f"Random glucose of {glucose:.0f} mg/dL is at or above {RANDOM_GLUCOSE_DIABETES}, "
            "which can indicate diabetes. Please confirm with a doctor."
        )

    return flags, diabetes, prediabetes and not diabetes


def assess_risk(
    bundle: dict,
    *,
    gender: str,
    age: float,
    hypertension: bool,
    heart_disease: bool,
    smoking_history: str,
    bmi: float,
    hba1c: float,
    glucose: float,
    fasting: bool = False,
) -> RiskAssessment:
    row = pd.DataFrame(
        [
            {
                "gender": gender,
                "smoking_history": smoking_history,
                "age": float(age),
                "hypertension": int(hypertension),
                "heart_disease": int(heart_disease),
                "bmi": float(bmi),
                "HbA1c_level": float(hba1c),
                "blood_glucose_level": float(glucose),
            }
        ]
    )[bundle["features"]]

    prob = float(bundle["pipeline"].predict_proba(row)[0][1])

    if prob >= bundle["decision_threshold"]:
        level = "High"
    elif prob >= bundle["screening_threshold"]:
        level = "Elevated"
    else:
        level = "Low"

    flags, in_diabetes, in_prediabetes = clinical_flags(hba1c, glucose, fasting)

    # Clinical criteria override the model upwards, never downwards.
    if in_diabetes:
        level = "High"
    elif in_prediabetes and level == "Low":
        level = "Elevated"

    return RiskAssessment(
        probability=prob,
        level=level,
        clinical_flags=flags,
        in_diabetes_range=in_diabetes,
        in_prediabetes_range=in_prediabetes,
    )


def get_metrics_summary(metrics: Optional[dict]) -> Optional[dict]:
    """Small, template-friendly view of models/metrics.json."""
    if not metrics:
        return None
    test = metrics["test"]
    at_t = test["at_decision_threshold"]
    return {
        "model": metrics["best_model"],
        "accuracy": round(at_t["accuracy"] * 100, 1),
        "roc_auc": round(test["roc_auc"], 3),
        "recall": round(at_t["recall_sensitivity"] * 100, 1),
        "screening_recall": round(test["at_screening_threshold"]["recall_sensitivity"] * 100, 1),
        "n_test": metrics["n_test"],
    }
