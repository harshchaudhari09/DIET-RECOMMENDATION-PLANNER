"""
Streamlit Chronic Disease Diet Recommendation Planner
Features added:
 - Sidebar navigation + theme
 - Improved UI cards and layout
 - Persistent history saved to local JSON (history.json)
 - PDF export of recommendations (FPDF)
 - Upload lab reports (CSV) and diet logs (CSV)
 - Optional ML model integration: upload a sklearn .pkl to run predictions on uploaded CSV
 - File uploads, download buttons, and simple charts

Run:
    pip install -r requirements.txt
    streamlit run streamlit_diet_app.py

requirements.txt (example):
streamlit
pandas
fpdf
scikit-learn
matplotlib

Note: This script writes a small history.json file next to the app. In a deployed environment, switch to a database.
"""

import streamlit as st
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
import json
import os
from fpdf import FPDF
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import pickle

# ------------------------------
# Data model & rulebase (same as earlier)
# ------------------------------
@dataclass
class UserProfile:
    age: Optional[int] = None
    sex: Optional[str] = None
    disease: Optional[str] = None
    allergies: Optional[List[str]] = None
    preferences: Optional[List[str]] = None
    calories: Optional[int] = None
    goal: Optional[str] = None
    medications: Optional[List[str]] = None

DISEASE_RULES = {
    "diabetes": {
        "pattern": "Low-carbohydrate with emphasis on fiber",
        "avoid_keywords": ["sugar", "sugary", "soda", "white bread", "pastry", "dessert"],
    },
    "hypertension": {
        "pattern": "DASH (low sodium, high potassium from safe sources)",
        "avoid_keywords": ["processed", "canned", "salted", "high-sodium"],
    },
    "celiac": {
        "pattern": "Gluten-free",
        "avoid_keywords": ["wheat", "barley", "rye", "seitan"],
    },
    "ckd": {
        "pattern": "Renal-friendly (monitor protein, potassium, phosphorus)",
        "avoid_keywords": ["banana", "potato", "avocado", "tomato"],
    },
    "hyperlipidemia": {
        "pattern": "Heart-healthy (Mediterranean / low saturated fat)",
        "avoid_keywords": ["fried", "butter", "cream", "processed meat"],
    },
    "ibs": {
        "pattern": "Low-FODMAP",
        "avoid_keywords": ["onion", "garlic", "beans", "wheat", "apple", "pear"],
    },
}

MEAL_TEMPLATES = {
    "Low-carbohydrate with emphasis on fiber": [
        "Grilled salmon, mixed greens, steamed broccoli",
        "Greek yogurt (unsweetened) with berries and chia seeds",
        "Turkey lettuce wraps with avocado and tomato",
    ],
    "DASH (low sodium, high potassium from safe sources)": [
        "Baked chicken breast, quinoa, steamed green beans",
        "Oatmeal topped with walnuts and blueberries (no added salt)",
        "Lentil soup (low-sodium) with a side salad",
    ],
    "Gluten-free": [
        "Quinoa bowl with roasted vegetables and chickpeas",
        "Rice noodles with tofu and vegetables (gluten-free soy sauce)",
        "Grilled shrimp, polenta, saut\u00e9ed spinach",
    ],
    "Renal-friendly (monitor protein, potassium, phosphorus)": [
        "Egg-white omelet with bell peppers and herbs",
        "White fish, cauliflower mash, steamed carrots",
        "Chicken salad with lettuce, cucumber, apples (light dressing)",
    ],
    "Heart-healthy (Mediterranean / low saturated fat)": [
        "Grilled sardines, farro, arugula salad with olive oil",
        "Chickpea and vegetable stew with brown rice",
        "Baked trout, roasted Brussels sprouts, sweet potato",
    ],
    "Low-FODMAP": [
        "Grilled chicken, rice, carrots",
        "Egg and spinach frittata (no onion/garlic)",
        "Zucchini noodles with tomato and basil",
    ],
}

# ------------------------------
# Helpers
# ------------------------------
HISTORY_FILE = "history.json"


def normalize_list(items: Optional[List[str]]) -> List[str]:
    if not items:
        return []
    return [str(x).strip().lower() for x in items if x and str(x).strip()]


def contains_keyword(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def filter_meals_by_allergies_and_preferences(meals: List[str], allergies: List[str], preferences: List[str]) -> List[str]:
    filtered = []
    allergy_keywords = [a.lower() for a in allergies]
    pref_keywords = [p.lower() for p in preferences]

    for meal in meals:
        m = meal.lower()
        skip = False
        for a in allergy_keywords:
            if a and re.search(r'\b' + re.escape(a) + r'\b', m):
                skip = True
                break
        if skip:
            continue

        if "vegetarian" in pref_keywords and re.search(r'\b(chicken|pork|beef|salmon|trout|shrimp|sardines|fish|turkey)\b', m):
            continue
        if "vegan" in pref_keywords and re.search(r'\b(egg|yogurt|cheese|salmon|shrimp|chicken|turkey)\b', m):
            continue
        if "gluten-free" in pref_keywords and re.search(r'\b(bread|pasta|wheat|barley|rye)\b', m):
            continue

        filtered.append(meal)

    return filtered or meals


def calculate_macros(calories: Optional[int], disease: Optional[str], goal: Optional[str]) -> Dict[str, Any]:
    if not calories or calories <= 0:
        calories = 2000

    carbs_pct = 0.50
    protein_pct = 0.20
    fat_pct = 0.30

    disease = (disease or "").lower()
    if "diabetes" in disease:
        carbs_pct = 0.30
        fat_pct = 0.35
        protein_pct = 0.35
    elif "hyperlipidemia" in disease:
        fat_pct = 0.25
    elif "ckd" in disease:
        protein_pct = 0.15

    if goal == "weight_loss":
        calories = max(1200, calories - 500)
    elif goal == "gain":
        calories += 300

    grams = lambda pct, is_fat: round((calories * pct) / (9 if is_fat else 4))

    return {
        "calories": calories,
        "carbohydrates_g": grams(carbs_pct, False),
        "protein_g": grams(protein_pct, False),
        "fat_g": grams(fat_pct, True),
        "note": "Macro targets are approximate."
    }


def generate_recommendation(profile: UserProfile) -> Dict[str, Any]:
    disease_key = (profile.disease or "").lower()
    rule = DISEASE_RULES.get(disease_key, {
        "pattern": "Balanced diet (whole grains, lean protein, fruits, vegetables)",
        "avoid_keywords": ["sugary", "fried", "processed"]
    })

    pattern = rule["pattern"]
    meals = MEAL_TEMPLATES.get(pattern, [
        "Oatmeal with fruit",
        "Grilled protein, vegetables, whole grain",
        "Salad with beans/tofu"
    ])

    allergies = normalize_list(profile.allergies)
    preferences = normalize_list(profile.preferences)

    meals = filter_meals_by_allergies_and_preferences(meals, allergies, preferences)
    meals = [m for m in meals if not contains_keyword(m, rule["avoid_keywords"])]

    macros = calculate_macros(profile.calories, profile.disease, profile.goal)

    warnings = []
    meds = normalize_list(profile.medications)
    if any("warfarin" in m for m in meds):
        warnings.append("Warfarin users should manage vitamin K intake consistently.")
    if any("metformin" in m for m in meds):
        warnings.append("Metformin may reduce B12 levels long-term.")
    if any("ace" in m for m in meds):
        warnings.append("ACE inhibitors may interact with high-potassium foods.")

    return {
        "pattern": pattern,
        "recommended_meals": meals,
        "macros": macros,
        "warnings": warnings,
        "disclaimer": "This is algorithmic and not medical advice."
    }

# ------------------------------
# Persistence helpers
# ------------------------------

def load_history() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(entry: Dict[str, Any]):
    h = load_history()
    h.insert(0, entry)
    # cap history to 50 entries
    h = h[:50]
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2)

# ------------------------------
# PDF export
# ------------------------------

def generate_pdf(profile: Dict[str, Any], rec: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, "Diet Recommendation", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 6, f"Patient: Age {profile.get('age')}  Sex: {profile.get('sex')}", ln=True)
    pdf.cell(0, 6, f"Disease: {profile.get('disease')}   Goal: {profile.get('goal')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "Recommended Pattern:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, rec.get('pattern', ''))
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "Meals:", ln=True)
    pdf.set_font("Arial", size=10)
    for m in rec.get('recommended_meals', []):
        pdf.multi_cell(0, 6, f"- {m}")

    pdf.ln(2)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 6, "Macro Targets:", ln=True)
    pdf.set_font("Arial", size=10)
    macros = rec.get('macros', {})
    for k, v in macros.items():
        pdf.cell(0, 6, f"{k}: {v}", ln=True)

    pdf.ln(2)
    if rec.get('warnings'):
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 6, "Warnings:", ln=True)
        pdf.set_font("Arial", size=10)
        for w in rec.get('warnings', []):
            pdf.multi_cell(0, 6, f"- {w}")

    pdf.ln(4)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, rec.get('disclaimer', ''))

    return pdf.output(dest='S').encode('latin-1')

# ------------------------------
# UI Layout
# ------------------------------

st.set_page_config(page_title="Diet Recommendation Planner", layout="wide")

with st.sidebar:
    st.header("Menu")
    page = st.radio("Go to", ["Planner", "History", "Upload / ML", "Settings"])
    st.markdown("---")
    st.write("Built for education — not a substitute for medical advice.")

if "loaded_model" not in st.session_state:
    st.session_state.loaded_model = None

# Planner page
if page == "Planner":
    st.title("🩺 Diet Recommendation Planner")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Profile")
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30)
        sex = st.selectbox("Sex", ["male", "female", "other"], index=0)
        disease = st.selectbox("Disease", ["none", "diabetes", "hypertension", "celiac", "ckd", "hyperlipidemia", "ibs"], index=0)
        calories = st.number_input("Daily Calorie Intake (optional)", min_value=0, step=50, value=2000)
        goal = st.selectbox("Goal", ["maintenance", "weight_loss", "gain"], index=0)

        allergies_txt = st.text_input("Allergies (comma-separated)")
        preferences_txt = st.text_input("Preferences (comma-separated, e.g. vegetarian, gluten-free)")
        meds_txt = st.text_input("Medications (comma-separated)")

        submitted = st.button("Generate Recommendation")

    with col2:
        st.subheader("Quick tips")
        st.info("Fill profile then click Generate. You can export PDF or save to history.")
        st.write("Upload lab reports or diet logs from the 'Upload / ML' page to visualize results.")

    if submitted:
        allergies = [x.strip() for x in allergies_txt.split(",") if x.strip()]
        preferences = [x.strip() for x in preferences_txt.split(",") if x.strip()]
        meds = [x.strip() for x in meds_txt.split(",") if x.strip()]

        profile = UserProfile(age=age, sex=sex, disease=disease if disease != 'none' else None,
                              allergies=allergies, preferences=preferences, calories=calories, goal=goal, medications=meds)

        rec = generate_recommendation(profile)

        # Show results in a two-column card style
        st.subheader("Recommendation")
        left, right = st.columns([2, 1])
        with left:
            st.markdown(f"**Pattern:** {rec['pattern']}")
            st.markdown("**Meals:**")
            for m in rec['recommended_meals']:
                st.write(f"- {m}")

        with right:
            st.markdown("**Macros**")
            st.json(rec['macros'])
            if rec['warnings']:
                for w in rec['warnings']:
                    st.warning(w)

        # Save to history
        entry = {"profile": asdict(profile), "recommendation": rec}
        save_history(entry)
        st.success("Saved to history")

        # PDF export
        pdf_bytes = generate_pdf(asdict(profile), rec)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="diet_recommendation.pdf">Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

# History page
elif page == "History":
    st.title("History")
    h = load_history()
    if not h:
        st.info("No history yet. Generate a recommendation in Planner to save entries.")
    else:
        for i, item in enumerate(h):
            with st.expander(f"Entry {i+1} — Age {item['profile'].get('age')} — {item['profile'].get('disease')}"):
                st.json(item)
                if st.button(f"Export PDF for entry {i+1}", key=f"pdf_{i}"):
                    pdf_bytes = generate_pdf(item['profile'], item['recommendation'])
                    st.download_button("Download PDF", data=pdf_bytes, file_name=f"recommendation_{i+1}.pdf")

# Upload / ML page
elif page == "Upload / ML":
    st.title("Upload & ML Tools")
    st.subheader("Upload lab report or diet log (CSV)")
    uploaded = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())

            st.markdown("### Simple visualization")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Choose numeric column to plot", numeric_cols)
                fig, ax = plt.subplots()
                ax.plot(df.index, df[col])
                ax.set_title(col)
                st.pyplot(fig)
            else:
                st.info("No numeric columns to plot.")

            # Save uploaded CSV to history folder optionally
            if st.button("Save uploaded CSV to history"):
                data_bytes = uploaded.getvalue()
                fname = f"upload_{int(pd.Timestamp.now().timestamp())}.csv"
                with open(fname, "wb") as f:
                    f.write(data_bytes)
                st.success(f"Saved as {fname}")

            # If model loaded, allow batch predictions
            if st.session_state.loaded_model is not None:
                if st.button("Run model on this CSV"):
                    model = st.session_state.loaded_model
                    try:
                        preds = model.predict(df.select_dtypes(include=['number']).fillna(0))
                        st.write("Predictions:")
                        st.write(preds)
                        # allow download
                        csv_out = df.copy()
                        csv_out['prediction'] = preds
                        csv_bytes = csv_out.to_csv(index=False).encode()
                        st.download_button("Download predictions CSV", csv_bytes, file_name="predictions.csv")
                    except Exception as e:
                        st.error(f"Model prediction failed: {e}")

        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    st.markdown("---")
    st.subheader("Optional: Upload sklearn model (.pkl)")
    model_file = st.file_uploader("Upload model pickle", type=["pkl"], key="model_uploader")
    if model_file:
        try:
            loaded = pickle.load(model_file)
            st.session_state.loaded_model = loaded
            st.success("Model loaded into session. Use 'Run model on this CSV' after uploading a CSV file.")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

# Settings
elif page == "Settings":
    st.title("Settings")
    st.write("Application settings and maintenance")
    if st.button("Clear history file"):
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
            st.success("History cleared")
        else:
            st.info("No history file found")

    st.markdown("### Export full history")
    if os.path.exists(HISTORY_FILE):
        if st.button("Download history.json"):
            with open(HISTORY_FILE, "rb") as f:
                st.download_button("Download", data=f, file_name=HISTORY_FILE)
    else:
        st.info("No history to export")

    st.markdown("---")
    st.write("Developer notes: this app writes small files locally. For production, switch to a secure DB and authentication.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© Diet Recommendation Planner — educational use only")
