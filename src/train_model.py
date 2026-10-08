"""
Trains and compares a few classifiers for diabetes risk prediction on
the Diabetes Prediction Dataset (~100k patient records), then saves
the best one.

Evaluation protocol (designed so the reported numbers are honest):

  1. De-duplicate, then hold out a stratified 20% test set that is
     never touched until the very end.
  2. Compare candidate models with 5-fold cross-validation on the
     training set only, and pick the winner by CV ROC-AUC. (The old
     version picked the winner by *test* score, which quietly leaks
     the test set into model selection.)
  3. Choose the "screening" threshold from out-of-fold training
     predictions -- again, no test data involved.
  4. Fit the winner on the full training set and evaluate it once on
     the held-out test set.

Run: python -m src.train_model
"""

import json

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
)
from src.data import load_data

# Probability at/above which we call someone "High risk".
DECISION_THRESHOLD = 0.5

# For the "Elevated risk" band we want a screening cut-off that catches
# most true cases: the highest threshold whose out-of-fold recall on the
# training set is still at least this value.
SCREENING_TARGET_RECALL = 0.85


def _one_hot_preprocessor():
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def _ordinal_preprocessor():
    # Tree models don't need scaling; ordinal codes + native categorical
    # support in HistGradientBoosting is faster than one-hot.
    return ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )


def candidate_models():
    n_cat = len(CATEGORICAL_FEATURES)
    return {
        "LogisticRegression": Pipeline(
            [
                ("prep", _one_hot_preprocessor()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("prep", _one_hot_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("prep", _ordinal_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=4,
                        learning_rate=0.1,
                        max_iter=300,
                        categorical_features=list(range(n_cat)),
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def pick_screening_threshold(y_true, proba, target_recall=SCREENING_TARGET_RECALL):
    """Highest threshold that still reaches `target_recall` on (y_true, proba)."""
    for t in np.arange(0.50, 0.01, -0.01):
        if recall_score(y_true, proba >= t) >= target_recall:
            return round(float(t), 2)
    return 0.05


def classification_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall_sensitivity": round(recall_score(y_true, pred), 4),
        "specificity": round(tn / (tn + fp), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    df = load_data()
    X = df[MODEL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"Rows after cleaning: {len(df):,}  (train {len(X_train):,} / test {len(X_test):,})")
    print(f"Diabetes prevalence: {y.mean():.1%}")

    # --- 1. Model comparison: 5-fold CV on the training set only ---
    print("\n=== Model comparison (5-fold CV on training set) ===")
    cv_results, oof_probas = {}, {}
    for name, pipeline in candidate_models().items():
        oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof_probas[name] = oof
        cv_results[name] = {
            "cv_roc_auc": round(roc_auc_score(y_train, oof), 4),
            "cv_accuracy": round(accuracy_score(y_train, oof >= DECISION_THRESHOLD), 4),
            "cv_balanced_accuracy": round(
                balanced_accuracy_score(y_train, oof >= DECISION_THRESHOLD), 4
            ),
        }
        r = cv_results[name]
        print(
            f"  {name:22s} ROC-AUC {r['cv_roc_auc']:.3f}   accuracy {r['cv_accuracy']:.3f}"
            f"   balanced acc {r['cv_balanced_accuracy']:.3f}"
        )

    best_name = max(cv_results, key=lambda n: cv_results[n]["cv_roc_auc"])
    screening_threshold = pick_screening_threshold(y_train, oof_probas[best_name])
    print(f"\nSelected: {best_name} (best CV ROC-AUC)")
    print(f"Screening threshold (>= {SCREENING_TARGET_RECALL:.0%} CV recall): {screening_threshold}")

    # --- 2. Fit on all training data, evaluate ONCE on the held-out test set ---
    best_pipeline = candidate_models()[best_name]
    best_pipeline.fit(X_train, y_train)
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    test_results = {
        "roc_auc": round(roc_auc_score(y_test, test_proba), 4),
        "majority_class_baseline_accuracy": round(baseline_acc, 4),
        "at_decision_threshold": classification_metrics(y_test, test_proba, DECISION_THRESHOLD),
        "at_screening_threshold": classification_metrics(y_test, test_proba, screening_threshold),
    }

    print("\n=== Held-out test set ===")
    print(f"  ROC-AUC: {test_results['roc_auc']:.3f}")
    print(f"  Majority-class baseline accuracy: {baseline_acc:.3f}")
    for key in ("at_decision_threshold", "at_screening_threshold"):
        m = test_results[key]
        print(
            f"  threshold {m['threshold']:.2f}: accuracy {m['accuracy']:.3f}  "
            f"balanced acc {m['balanced_accuracy']:.3f}  recall {m['recall_sensitivity']:.3f}  "
            f"precision {m['precision']:.3f}  specificity {m['specificity']:.3f}"
        )
    print()
    print(classification_report(y_test, test_proba >= DECISION_THRESHOLD, digits=3))

    # --- 3. Save the model together with everything needed to use it ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "features": MODEL_FEATURES,
        "decision_threshold": DECISION_THRESHOLD,
        "screening_threshold": screening_threshold,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "dataset": DATA_PATH.name,
        "n_rows_after_cleaning": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "prevalence": round(float(y.mean()), 4),
        "sklearn_version": sklearn.__version__,
        "cv_model_comparison": cv_results,
        "best_model": best_name,
        "test": test_results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
