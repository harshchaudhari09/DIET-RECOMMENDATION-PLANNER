import pytest

from src.diet_rules import (
    MIN_CALORIES,
    bmi_category,
    bp_category,
    calculate_bmi,
    estimate_bmr,
    healthy_weight_range,
    recommend_diet,
)

BASE = dict(
    weight_kg=70, height_cm=175, age=30, gender="Male", activity="moderate",
    systolic=115, diastolic=75, glycemic_control=False,
)


@pytest.mark.parametrize(
    "bmi,expected",
    [(17.0, "Underweight"), (18.5, "Normal"), (24.9, "Normal"), (25, "Overweight"), (30, "Obese")],
)
def test_bmi_category(bmi, expected):
    assert bmi_category(bmi) == expected


@pytest.mark.parametrize(
    "sys,dia,expected",
    [
        (115, 75, "Normal"),
        (125, 75, "Elevated"),
        (132, 75, "Stage 1 hypertension"),
        (118, 85, "Stage 1 hypertension"),
        (145, 85, "Stage 2 hypertension"),
        (185, 100, "Hypertensive crisis"),
    ],
)
def test_bp_category(sys, dia, expected):
    assert bp_category(sys, dia) == expected


def test_mifflin_st_jeor_known_values():
    # 10*70 + 6.25*175 - 5*30 = 1643.75
    assert estimate_bmr(70, 175, 30, "Male") == pytest.approx(1648.75)
    assert estimate_bmr(70, 175, 30, "Female") == pytest.approx(1482.75)


def test_healthy_weight_range():
    low, high = healthy_weight_range(175)
    assert calculate_bmi(low, 175) == pytest.approx(18.5, abs=0.05)
    assert calculate_bmi(high, 175) == pytest.approx(24.9, abs=0.05)


def test_balanced_plan_macros_add_up():
    plan = recommend_diet(**BASE)
    assert plan.label == "Balanced Diet"
    kcal = plan.protein_g * 4 + plan.carbs_g * 4 + plan.fat_g * 9
    assert kcal == pytest.approx(plan.daily_calories, rel=0.01)
    assert sum(m["calories"] for m in plan.meals.values()) == pytest.approx(
        plan.daily_calories, abs=3
    )


def test_glycemic_plan_lowers_carbs_and_sugar():
    normal = recommend_diet(**BASE)
    glycemic = recommend_diet(**{**BASE, "glycemic_control": True})
    assert glycemic.carbs_g < normal.carbs_g
    assert glycemic.added_sugar_g < normal.added_sugar_g
    assert glycemic.macro_percentages["carbs"] == 40
    assert "Diabetes-Friendly" in glycemic.label
    assert len(glycemic.meals) == 5  # smaller, more frequent meals


def test_high_bp_triggers_dash_limits():
    plan = recommend_diet(**{**BASE, "systolic": 145, "diastolic": 92})
    assert plan.sodium_mg == 1500
    assert "DASH" in plan.label


def test_obese_gets_deficit_but_never_below_floor():
    obese = recommend_diet(**{**BASE, "weight_kg": 110})
    assert "Weight-Loss" in obese.label

    tiny = recommend_diet(
        **{**BASE, "gender": "Female", "weight_kg": 85, "height_cm": 145,
           "age": 75, "activity": "sedentary"}
    )
    assert tiny.daily_calories >= MIN_CALORIES["Female"]


def test_warnings_for_crisis_and_diabetes_range():
    plan = recommend_diet(**{**BASE, "systolic": 190, "diastolic": 125,
                             "glycemic_control": True, "in_diabetes_range": True})
    assert len(plan.warnings) == 2
