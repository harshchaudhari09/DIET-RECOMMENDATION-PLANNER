# CuraDiet — Chronic Disease Risk & Diet Recommendation Planner

An end-to-end ML project that predicts diabetes risk from a patient's
labs and vitals, then generates a personalized daily calorie, macro and
meal plan using a rule-based engine grounded in clinical guidelines
(ADA, ACC/AHA, DASH).

**Held-out test results (19,226 unseen patients):** 97.2% accuracy ·
0.978 ROC-AUC · 98.0% precision. A screening mode catches 84.7% of
diabetes cases at 93.7% accuracy.

## What it does

1. **Diabetes risk model.** A HistGradientBoosting classifier trained on
   ~96k de-duplicated patient records. It uses age, sex, BMI, HbA1c,
   blood glucose, hypertension, heart disease and smoking history.
2. **Clinical safety layer.** ADA diagnostic cut-offs (HbA1c ≥ 6.5%,
   fasting glucose ≥ 126, random glucose ≥ 200 mg/dL, plus the
   prediabetes ranges) are checked independently of the model. They can
   only raise the risk level, never lower it. Risk is shown as
   **Low / Elevated / High**.
3. **Diet engine (deliberately not ML).** It works in four steps:
   - Mifflin-St Jeor BMR × activity factor gives TDEE.
   - A calorie deficit or surplus is applied by BMI, with safe minimum
     floors.
   - The macro split moves to a lower-carb ADA style when glucose control
     matters.
   - Sodium and saturated fat are capped DASH-style when blood pressure
     is raised.

   The plan also includes fibre, sodium, sugar and saturated-fat
   targets, a per-meal split and food suggestions.
4. **Flask app.** It validates every input with friendly errors and
   shows the risk, the plan, and an explanation of *why* each number was
   chosen.

## Results

Protocol: de-duplicate → stratified 80/20 split → compare models with
5-fold CV on the **training set only** → pick the winner by CV ROC-AUC →
choose the screening threshold from out-of-fold predictions → evaluate
**once** on the untouched test set. See `src/train_model.py`.

**Model comparison (5-fold CV, training set)**

| Model | CV ROC-AUC | CV Accuracy |
|---|---|---|
| Logistic Regression | 0.962 | 95.9% |
| Random Forest | 0.970 | 97.0% |
| **HistGradientBoosting (selected)** | **0.979** | **97.1%** |

**Held-out test set (n = 19,226)**

| Operating point | Accuracy | Precision | Recall | Specificity | Balanced acc. |
|---|---|---|---|---|---|
| "High risk" (p ≥ 0.50) | **97.2%** | 98.0% | 69.6% | 99.9% | 84.7% |
| "Elevated" screening (p ≥ 0.16) | **93.7%** | 59.9% | 84.7% | 94.5% | 89.6% |

ROC-AUC on the test set is **0.978**.

**Reading these numbers honestly:** only 8.8% of patients have
diabetes, so "always predict no" already scores 91.2%. Accuracy alone
is misleading on data this imbalanced, which is why ROC-AUC, recall and
balanced accuracy are reported too. The two thresholds exist because a
screening tool should miss as few real cases as possible. The
"Elevated" band trades some precision for much higher recall.

## Why the dataset changed (v1 → v2)

v1 used the **Pima Indians Diabetes Dataset** and reached about 74%
test accuracy. `python -m src.benchmark_pima` shows that limit comes
from the data, not the model. Three very different models all plateau
in the same place:

| Model (5-fold CV on Pima) | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 77.2% | 0.837 |
| Random Forest | 76.6% | 0.835 |
| HistGradientBoosting | 75.6% | 0.821 |

Pima has only 768 rows, many zero-coded missing values, and only women
aged 21+ of one heritage. Repos that claim 90%+ on Pima usually leak
data, for example by oversampling before the split or tuning on the
test set. v2 switched to a larger dataset that includes HbA1c. HbA1c is
the strongest clinical marker for diabetes, and the switch also removed
the female-only limitation.

## Bugs fixed from v1

- **Crash on empty or invalid input.** `float("")` raised a 500 error.
  All fields are now validated with clear messages, and entered values
  are kept.
- **Relative paths.** The app only worked when launched from the project
  root. All paths now resolve from `src/config.py`.
- **Test-set leakage in model selection.** v1 picked the "best" model by
  test score. v2 selects by CV on the training set.
- **Imputation leakage.** v1 computed medians on the full dataset before
  splitting. The new dataset needs no imputation, and the Pima benchmark
  imputes inside each CV fold.
- **Wrong blood-pressure categories.** 130/80 was labelled "Elevated".
  It now follows ACC/AHA 2017: Normal, Elevated, Stage 1, Stage 2,
  Crisis.
- **Hidden default inputs.** v1 silently filled insulin, skin thickness
  and pedigree with made-up constants. v2 uses only inputs the user
  actually enters.
- **Unsafe calorie targets.** Small or elderly users could get very low
  targets. Floors are now 1,200 kcal (female) and 1,500 kcal (male).
- **Pickled-model version drift.** The model is saved with its
  scikit-learn version, and the app warns on a mismatch.
- **Startup and config issues.** The app no longer crashes on import if
  the model is missing. `debug=True` is off by default. The README port
  now matches the code.

## Known limitations

- The dataset comes from a public Kaggle release with limited
  documentation about how it was collected. Treat results as a
  demonstration, not a validated clinical tool.
- The model needs an HbA1c value from a blood test. Without HbA1c,
  accuracy drops sharply (test ROC-AUC about 0.93).
- Recall at the 0.5 threshold is 70%. That is why the app uses a
  lower screening threshold plus hard ADA cut-offs rather than one
  yes/no answer.
- The diet engine encodes general guidance, not individualized medical
  advice.

## Project structure

```
├── data/
│   ├── diabetes_prediction_dataset.csv  # ~100k records (training data)
│   └── pima_diabetes.csv                # v1 dataset, kept for the benchmark
├── src/
│   ├── config.py           # paths, feature lists, input ranges
│   ├── data.py             # loading + cleaning (dedupe, drop "Other")
│   ├── train_model.py      # CV model comparison, threshold, test evaluation
│   ├── risk.py             # model inference + ADA clinical cut-offs
│   ├── diet_rules.py       # BMR/TDEE + ADA/DASH diet engine
│   ├── validation.py       # form parsing + validation
│   └── benchmark_pima.py   # shows the Pima accuracy ceiling
├── tests/                  # 32 pytest tests (rules, validation, model, app)
├── models/                 # saved model + metrics.json (generated)
├── templates/, static/
└── app.py                  # Flask app
```

## Running it

```bash
pip install -r requirements.txt
python -m src.train_model      # ~15 s: trains, compares, saves model + metrics
python -m pytest               # run the test suite
python app.py                  # http://127.0.0.1:5000
```

## Possible next steps

- Probability calibration (isotonic) plus a reliability plot
- SHAP explanations of which factors drove each prediction
- Persist submissions to track progress over time
- Deploy (Render/Railway) and add a live demo link
