"""
Diet recommendation engine.

This is intentionally NOT a machine learning model. There's no public
dataset that reliably maps "patient profile -> correct diet plan" with
real signal (we checked -- see README "Design notes"), so instead this
encodes the same logic a dietitian actually uses:

  1. Estimate daily calorie needs with the Mifflin-St Jeor equation
     (the most widely validated BMR formula) x an activity multiplier
     to get TDEE (Total Daily Energy Expenditure).
  2. Adjust that target and the macro split based on BMI category,
     diabetes risk, and blood pressure, following standard ADA
     (American Diabetes Association), ACC/AHA blood-pressure and DASH
     diet guidance.
  3. Add daily limits (sodium, added sugar, saturated fat), a fibre
     target, a per-meal calorie split and concrete food suggestions.

Everything here is a pure function, so it's covered by unit tests in
tests/test_diet_rules.py.
"""

from dataclasses import dataclass, field

from src.config import ACTIVITY_MULTIPLIERS

# Never recommend intakes below these without medical supervision
MIN_CALORIES = {"Female": 1200, "Male": 1500}

# Calories per gram
KCAL_PER_G = {"protein": 4, "carbs": 4, "fat": 9}

# Share of daily calories per meal
MEAL_SPLIT_STANDARD = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.30, "Snack": 0.10}
# Smaller, more even meals help keep blood glucose steady
MEAL_SPLIT_GLYCEMIC = {
    "Breakfast": 0.25,
    "Lunch": 0.30,
    "Dinner": 0.25,
    "Mid-morning snack": 0.10,
    "Afternoon snack": 0.10,
}


@dataclass
class DietPlan:
    label: str
    daily_calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    fiber_g: int
    sodium_mg: int
    added_sugar_g: int
    saturated_fat_g: int
    bmi: float
    bmi_category: str
    bp_category: str
    healthy_weight_range_kg: tuple
    meals: dict = field(default_factory=dict)
    foods_to_emphasize: list = field(default_factory=list)
    foods_to_limit: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def macro_percentages(self) -> dict:
        kcal = {
            "protein": self.protein_g * KCAL_PER_G["protein"],
            "carbs": self.carbs_g * KCAL_PER_G["carbs"],
            "fat": self.fat_g * KCAL_PER_G["fat"],
        }
        total = sum(kcal.values())
        return {k: round(v / total * 100) for k, v in kcal.items()}


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    return weight_kg / ((height_cm / 100) ** 2)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def bp_category(systolic: int, diastolic: int) -> str:
    """2017 ACC/AHA blood pressure categories."""
    if systolic >= 180 or diastolic >= 120:
        return "Hypertensive crisis"
    if systolic >= 140 or diastolic >= 90:
        return "Stage 2 hypertension"
    if systolic >= 130 or diastolic >= 80:
        return "Stage 1 hypertension"
    if systolic >= 120:
        return "Elevated"
    return "Normal"


def healthy_weight_range(height_cm: float) -> tuple:
    """Weight range (kg) that corresponds to BMI 18.5-24.9 at this height."""
    h2 = (height_cm / 100) ** 2
    return round(18.5 * h2, 1), round(24.9 * h2, 1)


def estimate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Mifflin-St Jeor basal metabolic rate."""
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return base + 5 if gender == "Male" else base - 161


def estimate_tdee(weight_kg: float, height_cm: float, age: int, gender: str, activity: str) -> float:
    """BMR scaled by activity level to get TDEE."""
    return estimate_bmr(weight_kg, height_cm, age, gender) * ACTIVITY_MULTIPLIERS[activity]


def recommend_diet(
    weight_kg: float,
    height_cm: float,
    age: int,
    gender: str,
    activity: str,
    systolic: int,
    diastolic: int,
    glycemic_control: bool,
    in_diabetes_range: bool = False,
) -> DietPlan:
    """
    glycemic_control: True when diabetes risk is Elevated/High or labs
    are in the prediabetes/diabetes range -> lower-carb ADA-style plan.
    """
    bmi = calculate_bmi(weight_kg, height_cm)
    bmi_cat = bmi_category(bmi)
    bp_cat = bp_category(systolic, diastolic)
    bp_concern = bp_cat != "Normal"

    tdee = estimate_tdee(weight_kg, height_cm, age, gender, activity)

    notes, warnings, label_parts = [], [], []

    # --- Calorie target: deficit for overweight/obese, maintenance otherwise ---
    if bmi_cat == "Obese":
        calories = tdee * 0.80
        notes.append("20% calorie deficit for gradual, sustainable weight loss (~0.5 kg/week).")
        label_parts.append("Weight-Loss")
    elif bmi_cat == "Overweight":
        calories = tdee * 0.90
        notes.append("10% calorie deficit to move toward a healthier BMI.")
        label_parts.append("Weight-Loss")
    elif bmi_cat == "Underweight":
        calories = tdee * 1.10
        notes.append("10% calorie surplus to support healthy weight gain.")
        label_parts.append("Weight-Gain")
    else:
        calories = tdee
        notes.append("Calories set at maintenance (TDEE) for your current weight.")

    floor = MIN_CALORIES.get(gender, 1200)
    if calories < floor:
        calories = floor
        notes.append(
            f"Raised to {floor} kcal: very-low-calorie diets should only be followed "
            "under medical supervision."
        )

    # --- Macro split: lower-carb if blood glucose needs managing (ADA guidance) ---
    if glycemic_control:
        carb_pct, protein_pct, fat_pct = 0.40, 0.25, 0.35
        label_parts.append("Diabetes-Friendly Low-Carb")
        notes.append(
            "Carbohydrates reduced to ~40% of calories, with more protein and healthy fats, "
            "to help manage blood glucose (ADA guidance). Spread carbs evenly across meals "
            "and pair them with protein or fibre."
        )
        added_sugar_pct = 0.05
    else:
        carb_pct, protein_pct, fat_pct = 0.50, 0.20, 0.30
        added_sugar_pct = 0.10

    # --- Sodium + saturated fat limits if blood pressure is raised (DASH / AHA) ---
    if bp_concern:
        sodium_mg = 1500
        sat_fat_pct = 0.06
        label_parts.append("Low-Sodium DASH")
        notes.append(
            f"Blood pressure is {bp_cat.lower()}: sodium capped at 1,500 mg/day and saturated "
            "fat at ~6% of calories, with plenty of potassium-rich fruit and vegetables (DASH)."
        )
    else:
        sodium_mg = 2300
        sat_fat_pct = 0.10

    if not label_parts or label_parts == ["Weight-Gain"]:
        label_parts.append("Balanced")

    if bp_cat == "Hypertensive crisis":
        warnings.append(
            "Your blood pressure reading is in the hypertensive-crisis range (≥180/120). "
            "Re-measure and seek medical care promptly, especially if you have symptoms."
        )
    if in_diabetes_range:
        warnings.append(
            "Your lab values are in the diabetes range. This plan is general guidance -- "
            "please work with a doctor or registered dietitian, especially if you take "
            "insulin or glucose-lowering medication (risk of low blood sugar)."
        )

    calories = round(calories)
    protein_g = round(calories * protein_pct / KCAL_PER_G["protein"])
    carbs_g = round(calories * carb_pct / KCAL_PER_G["carbs"])
    fat_g = round(calories * fat_pct / KCAL_PER_G["fat"])

    # Dietary Guidelines for Americans: 14 g fibre per 1,000 kcal
    fiber_g = round(calories / 1000 * 14)
    added_sugar_g = round(calories * added_sugar_pct / KCAL_PER_G["carbs"])
    saturated_fat_g = round(calories * sat_fat_pct / KCAL_PER_G["fat"])

    split = MEAL_SPLIT_GLYCEMIC if glycemic_control else MEAL_SPLIT_STANDARD
    meals = {
        name: {
            "calories": round(calories * share),
            "carbs_g": round(carbs_g * share),
            "protein_g": round(protein_g * share),
        }
        for name, share in split.items()
    }

    emphasize, limit = _food_guidance(glycemic_control, bp_concern, bmi_cat)

    low, high = healthy_weight_range(height_cm)
    notes.append(
        f"BMI {bmi:.1f} ({bmi_cat}). A healthy weight for your height is about "
        f"{low:.0f}–{high:.0f} kg."
    )

    return DietPlan(
        label=" + ".join(label_parts) + " Diet",
        daily_calories=calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        fiber_g=fiber_g,
        sodium_mg=sodium_mg,
        added_sugar_g=added_sugar_g,
        saturated_fat_g=saturated_fat_g,
        bmi=round(bmi, 1),
        bmi_category=bmi_cat,
        bp_category=bp_cat,
        healthy_weight_range_kg=(low, high),
        meals=meals,
        foods_to_emphasize=emphasize,
        foods_to_limit=limit,
        notes=notes,
        warnings=warnings,
    )


def _food_guidance(glycemic_control: bool, bp_concern: bool, bmi_cat: str):
    emphasize = [
        "Non-starchy vegetables (leafy greens, broccoli, cauliflower, peppers) — half your plate",
        "Lean protein: dal, chickpeas, tofu, paneer (moderate), eggs, fish, chicken",
        "Healthy fats: nuts, seeds, olive/mustard oil, avocado",
    ]
    limit = ["Deep-fried foods and ultra-processed snacks"]

    if glycemic_control:
        emphasize += [
            "Low-GI whole grains in measured portions: oats, millets (jowar, bajra, ragi), "
            "brown rice, whole-wheat roti",
            "Whole fruit instead of juice — berries, guava, apple, pear",
        ]
        limit += [
            "Sugary drinks, sweets, and fruit juice",
            "Refined carbs: white bread, maida, large portions of white rice",
        ]
    else:
        emphasize.append("Whole grains: oats, brown rice, whole-wheat roti, millets")
        limit.append("Sugary drinks and sweets")

    if bp_concern:
        emphasize += [
            "Potassium-rich foods: bananas, spinach, sweet potato, beans, yoghurt",
            "Herbs, lemon and spices instead of extra salt",
        ]
        limit += [
            "Pickles, papad, chips, instant noodles and other salty packaged foods",
            "Processed meats and restaurant/takeaway meals (often very high in sodium)",
        ]

    if bmi_cat == "Underweight":
        emphasize.append("Energy-dense healthy foods: nut butters, dried fruit, full-fat dairy")
    elif bmi_cat in ("Overweight", "Obese"):
        limit.append("Calorie-dense extras: ghee/butter in large amounts, creamy gravies")

    return emphasize, limit
