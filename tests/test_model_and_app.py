"""
Integration tests: these need the trained model in models/
(run `python -m src.train_model` first).
"""

import json

import pytest

from src.config import METRICS_PATH, MODEL_PATH
from src.risk import assess_risk, clinical_flags, load_model_bundle

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="model not trained -- run python -m src.train_model"
)


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle()


@pytest.fixture(scope="module")
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_held_out_accuracy_is_at_least_90_percent():
    metrics = json.loads(METRICS_PATH.read_text())
    test = metrics["test"]
    assert test["at_decision_threshold"]["accuracy"] >= 0.90
    assert test["at_screening_threshold"]["accuracy"] >= 0.90
    assert test["roc_auc"] >= 0.95
    # Must clearly beat "always predict no diabetes"
    assert test["at_decision_threshold"]["accuracy"] > test["majority_class_baseline_accuracy"]


PROFILE = dict(gender="Female", age=35, hypertension=False, heart_disease=False,
               smoking_history="never", bmi=22.0)


def test_healthy_profile_is_low_risk(bundle):
    risk = assess_risk(bundle, **PROFILE, hba1c=5.0, glucose=90, fasting=True)
    assert risk.level == "Low"
    assert risk.probability < 0.05


def test_diabetic_labs_are_high_risk(bundle):
    risk = assess_risk(bundle, **{**PROFILE, "age": 60, "bmi": 33}, hba1c=8.2, glucose=240)
    assert risk.level == "High"
    assert risk.in_diabetes_range
    assert risk.probability > 0.9


def test_clinical_cutoffs_override_model_upwards(bundle):
    # Fasting glucose 130 is diabetic by ADA criteria even if the model is unsure
    risk = assess_risk(bundle, **PROFILE, hba1c=5.5, glucose=130, fasting=True)
    assert risk.level == "High"


def test_prediabetes_flag():
    flags, diabetes, prediabetes = clinical_flags(hba1c=6.0, glucose=95, fasting=True)
    assert prediabetes and not diabetes and len(flags) == 1


FORM = {
    "age": "45", "gender": "Male", "height_cm": "175", "weight_kg": "90",
    "activity": "moderate", "smoking_history": "former", "glucose": "160",
    "hba1c": "6.2", "systolic": "135", "diastolic": "85",
}


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Get my plan" in r.data


def test_predict_happy_path(client):
    r = client.post("/predict", data=FORM)
    assert r.status_code == 200
    assert b"Diabetes risk:" in r.data
    assert b"kcal / day" in r.data


def test_predict_bad_input_returns_form_with_errors(client):
    r = client.post("/predict", data={**FORM, "height_cm": ""})
    assert r.status_code == 400
    assert b"Height is required." in r.data
    assert b'value="90"' in r.data  # previously entered values are kept


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_get_predict_redirects_to_form(client):
    r = client.get("/predict")
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/")


def test_example_link_prefills_form(client):
    r = client.get("/?example=1")
    assert r.status_code == 200
    assert b'value="6.1"' in r.data
    assert b'value="Male" selected' in r.data
