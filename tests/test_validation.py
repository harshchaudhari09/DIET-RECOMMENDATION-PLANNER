from src.validation import parse_profile

VALID = {
    "age": "45", "gender": "Male", "height_cm": "175", "weight_kg": "90",
    "activity": "moderate", "smoking_history": "never", "glucose": "140",
    "hba1c": "6.0", "systolic": "135", "diastolic": "85",
}


def test_valid_form_parses():
    profile, errors = parse_profile(VALID)
    assert errors == []
    assert profile["age"] == 45
    assert profile["hba1c"] == 6.0
    assert profile["glucose_fasting"] is False


def test_empty_field_is_an_error_not_a_crash():
    _, errors = parse_profile({**VALID, "height_cm": ""})
    assert errors == ["Height is required."]


def test_non_numeric_and_out_of_range():
    _, errors = parse_profile({**VALID, "glucose": "abc", "hba1c": "40"})
    assert "Blood glucose must be a number." in errors
    assert any("HbA1c must be between" in e for e in errors)


def test_mmol_glucose_is_rejected_as_out_of_range():
    _, errors = parse_profile({**VALID, "glucose": "7.8"})
    assert any("Blood glucose must be between" in e for e in errors)


def test_invalid_choice_and_swapped_bp():
    _, errors = parse_profile({**VALID, "activity": "extreme", "systolic": "80", "diastolic": "120"})
    assert "Please choose a valid activity level." in errors
    assert any("swapped" in e for e in errors)


def test_checkboxes():
    profile, _ = parse_profile({**VALID, "glucose_fasting": "on", "heart_disease": "on"})
    assert profile["glucose_fasting"] and profile["heart_disease"]
    assert not profile["hypertension_dx"]
