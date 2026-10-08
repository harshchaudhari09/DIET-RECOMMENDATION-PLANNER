"""
Form parsing + validation for the web app.

The old app called float(form["..."]) directly, so an empty or
non-numeric field crashed the request with a 500 error. This turns
the raw form into clean typed values plus a list of friendly error
messages instead.
"""

from src.config import ACTIVITY_MULTIPLIERS, GENDERS, INPUT_RANGES, SMOKING_OPTIONS


def _parse_number(form, key, errors, integer=False):
    lo, hi, label = INPUT_RANGES[key]
    raw = (form.get(key) or "").strip()
    if raw == "":
        errors.append(f"{label} is required.")
        return None
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{label} must be a number.")
        return None
    if not lo <= value <= hi:
        errors.append(f"{label} must be between {lo:g} and {hi:g}.")
        return None
    return int(round(value)) if integer else value


def _parse_choice(form, key, options, label, errors):
    value = form.get(key)
    if value not in options:
        errors.append(f"Please choose a valid {label}.")
        return None
    return value


def _checkbox(form, key) -> bool:
    return form.get(key) in ("on", "1", "true", "yes")


def parse_profile(form):
    """Returns (profile_dict, errors). profile is only usable if errors is empty."""
    errors = []
    profile = {
        "age": _parse_number(form, "age", errors, integer=True),
        "gender": _parse_choice(form, "gender", GENDERS, "gender", errors),
        "height_cm": _parse_number(form, "height_cm", errors),
        "weight_kg": _parse_number(form, "weight_kg", errors),
        "activity": _parse_choice(form, "activity", ACTIVITY_MULTIPLIERS, "activity level", errors),
        "smoking_history": _parse_choice(
            form, "smoking_history", SMOKING_OPTIONS, "smoking history", errors
        ),
        "glucose": _parse_number(form, "glucose", errors),
        "glucose_fasting": _checkbox(form, "glucose_fasting"),
        "hba1c": _parse_number(form, "hba1c", errors),
        "systolic": _parse_number(form, "systolic", errors, integer=True),
        "diastolic": _parse_number(form, "diastolic", errors, integer=True),
        "hypertension_dx": _checkbox(form, "hypertension_dx"),
        "heart_disease": _checkbox(form, "heart_disease"),
    }

    if (
        profile["systolic"] is not None
        and profile["diastolic"] is not None
        and profile["diastolic"] >= profile["systolic"]
    ):
        errors.append(
            "Diastolic (lower) blood pressure should be less than systolic (upper) — "
            "check the two values aren't swapped."
        )

    return profile, errors
