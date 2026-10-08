"""
Loading + cleaning for the Diabetes Prediction Dataset.

Kept separate from training so the cleaning rules are easy to read
and to unit-test.
"""

import pandas as pd

from src.config import DATA_PATH, MODEL_FEATURES, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Only 18 of 100k rows use gender "Other" -- far too few for the
    # model to learn anything about them, and the diet engine's BMR
    # formula needs a sex-specific constant anyway.
    df = df[df["gender"].isin(["Female", "Male"])]

    # ~3.8k rows are exact duplicates. Dropping them BEFORE the
    # train/test split stops identical records landing on both sides,
    # which would inflate test accuracy.
    df = df.drop_duplicates().reset_index(drop=True)

    return df[MODEL_FEATURES + [TARGET]]
