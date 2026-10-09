"""
Flask web app: user fills in a health profile, we run the trained
diabetes risk model (plus ADA clinical cut-offs) to estimate risk,
then feed that into the rule-based diet engine to produce a
personalized calorie, macro and meal plan.

Run: python app.py   (then open http://127.0.0.1:5000)
"""

import json
import os

from flask import Flask, redirect, render_template, request, url_for

from src.config import ACTIVITY_MULTIPLIERS, GENDERS, METRICS_PATH, SMOKING_OPTIONS
from src.diet_rules import calculate_bmi, recommend_diet
from src.risk import ModelNotTrainedError, assess_risk, get_metrics_summary, load_model_bundle
from src.validation import parse_profile

app = Flask(__name__)

# Loaded once at startup, not per-request. If the model hasn't been
# trained yet the app still starts and shows a clear message instead
# of crashing on import.
try:
    model_bundle = load_model_bundle()
    startup_error = None
except ModelNotTrainedError as e:
    model_bundle = None
    startup_error = str(e)

try:
    with open(METRICS_PATH) as f:
        model_metrics = get_metrics_summary(json.load(f))
except (FileNotFoundError, KeyError, json.JSONDecodeError):
    model_metrics = None


def render_form(values=None, errors=None, status=200):
    return (
        render_template(
            "index.html",
            activities=ACTIVITY_MULTIPLIERS.keys(),
            genders=GENDERS,
            smoking_options=SMOKING_OPTIONS,
            values=values or {},
            errors=errors or [],
            startup_error=startup_error,
            metrics=model_metrics,
        ),
        status,
    )


# Pre-filled demo profile so visitors can try the app in two clicks
# (open /?example=1, then press "Get my plan").
EXAMPLE_PROFILE = {
    "age": "52",
    "gender": "Male",
    "height_cm": "170",
    "weight_kg": "92",
    "activity": "light",
    "smoking_history": "former",
    "hba1c": "6.1",
    "glucose": "165",
    "systolic": "138",
    "diastolic": "88",
}


@app.route("/", methods=["GET"])
def home():
    return render_form(EXAMPLE_PROFILE if request.args.get("example") else None)


@app.route("/healthz", methods=["GET"])
def healthz():
    # Used by the hosting platform's health check
    if model_bundle is None:
        return {"status": "error", "detail": startup_error}, 503
    return {"status": "ok", "model": model_bundle["model_name"]}


@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Someone opening /predict directly (or a shared link) gets the form
    # instead of a "405 Method Not Allowed" page.
    if request.method == "GET":
        return redirect(url_for("home"))
    if model_bundle is None:
        return render_form(request.form, [startup_error], status=503)

    profile, errors = parse_profile(request.form)
    if errors:
        return render_form(request.form, errors, status=400)

    bmi = calculate_bmi(profile["weight_kg"], profile["height_cm"])

    # The dataset's "hypertension" column is a diagnosis, so we count it as
    # present if the user says they've been diagnosed OR today's reading is
    # in the stage 2 range (>= 140/90).
    hypertension = (
        profile["hypertension_dx"] or profile["systolic"] >= 140 or profile["diastolic"] >= 90
    )

    risk = assess_risk(
        model_bundle,
        gender=profile["gender"],
        age=profile["age"],
        hypertension=hypertension,
        heart_disease=profile["heart_disease"],
        smoking_history=profile["smoking_history"],
        bmi=bmi,
        hba1c=profile["hba1c"],
        glucose=profile["glucose"],
        fasting=profile["glucose_fasting"],
    )

    diet = recommend_diet(
        weight_kg=profile["weight_kg"],
        height_cm=profile["height_cm"],
        age=profile["age"],
        gender=profile["gender"],
        activity=profile["activity"],
        systolic=profile["systolic"],
        diastolic=profile["diastolic"],
        glycemic_control=risk.needs_glycemic_diet,
        in_diabetes_range=risk.in_diabetes_range,
    )

    return render_template(
        "result.html",
        risk=risk,
        risk_prob=round(risk.probability * 100, 1),
        diet=diet,
        metrics=model_metrics,
    )


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        port=int(os.environ.get("PORT", 5000)),
    )
