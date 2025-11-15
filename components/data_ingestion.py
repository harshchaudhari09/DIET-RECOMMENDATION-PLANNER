from components.data_ingestion import DataIngestionConfig, run_ingestion
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

#!/usr/bin/env python3
"""
Data ingestion utilities for CHRONIC_DISEASE_DIET_RECOMMENDATION_PLANNER.

Provides:
- reading CSV/JSON (or directory of CSVs)
- basic schema validation
- lightweight preprocessing (dedupe, missing value fills)
- train/test split and saving

Usage:
    cfg = DataIngestionConfig(input_path="data/raw.csv", output_dir="data/processed", target_column="label")
    run_ingestion(cfg)
"""




@dataclass
class DataIngestionConfig:
        input_path: str
        output_dir: str = "data/processed"
        test_size: float = 0.2
        random_state: int = 42
        target_column: Optional[str] = None
        required_columns: Optional[List[str]] = None
        allowed_extensions: Tuple[str, ...] = field(default_factory=lambda: (".csv", ".json"))
        read_csv_kwargs: dict = field(default_factory=dict)


def _read_single_file(path: str, read_csv_kwargs: dict) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
                return pd.read_csv(path, **read_csv_kwargs)
        if ext == ".json":
                return pd.read_json(path)
        raise ValueError(f"Unsupported file extension: {ext} for file {path}")


def read_input(path: str, allowed_extensions: Iterable[str] = (".csv", ".json"), read_csv_kwargs: dict = None) -> pd.DataFrame:
        """
        Read a file or a directory (concatenate CSVs/JSONs). Returns a DataFrame.
        """
        read_csv_kwargs = read_csv_kwargs or {}
        if os.path.isdir(path):
                frames = []
                for ext in allowed_extensions:
                        for fname in sorted(glob(os.path.join(path, f"*{ext}"))):
                                frames.append(_read_single_file(fname, read_csv_kwargs))
                if not frames:
                        raise FileNotFoundError(f"No files with extensions {allowed_extensions} found in directory: {path}")
                return pd.concat(frames, ignore_index=True)
        elif os.path.isfile(path):
                return _read_single_file(path, read_csv_kwargs)
        else:
                raise FileNotFoundError(f"Input path not found: {path}")


def validate_schema(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate presence of required columns. Returns (is_valid, missing_columns).
        """
        if not required_columns:
                return True, []
        missing = [c for c in required_columns if c not in df.columns]
        return (len(missing) == 0), missing


def _fill_numeric(series: pd.Series) -> pd.Series:
        if series.dtype.kind in "biufc":  # numeric types
                if series.isna().any():
                        return series.fillna(series.median())
        return series


def _fill_categorical(series: pd.Series) -> pd.Series:
        if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
                if series.isna().any():
                        mode = series.mode(dropna=True)
                        fill = mode.iloc[0] if not mode.empty else ""
                        return series.fillna(fill)
        return series


def preprocess(df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Basic preprocessing:
        - strip column names
        - drop exact duplicate rows
        - fill numeric with median, categorical with mode/empty string
        - remove rows missing target_column (if provided)
        """
        # normalize column names
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # drop exact duplicates
        df = df.drop_duplicates(ignore_index=True)

        # remove completely empty columns
        df = df.dropna(axis=1, how="all")

        # remove rows where target is missing (if target provided)
        if target_column:
                if target_column not in df.columns:
                        raise KeyError(f"target_column '{target_column}' not found in data")
                df = df.dropna(subset=[target_column]).reset_index(drop=True)

        # Fill per-column
        for col in df.columns:
                s = df[col]
                if pd.api.types.is_numeric_dtype(s):
                        df[col] = _fill_numeric(s)
                else:
                        df[col] = _fill_categorical(s)

        return df


def split_and_save(df: pd.DataFrame, config: DataIngestionConfig) -> Tuple[str, str]:
        """
        Split DataFrame into train/test and save CSVs to output_dir.
        Returns (train_path, test_path)
        """
        os.makedirs(config.output_dir, exist_ok=True)
        if config.target_column and config.target_column in df.columns:
                X = df.drop(columns=[config.target_column])
                y = df[[config.target_column]]
                X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=config.test_size, random_state=config.random_state, shuffle=True
                )
                train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
                test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
        else:
                train_df, test_df = train_test_split(df, test_size=config.test_size, random_state=config.random_state, shuffle=True)

        train_path = os.path.join(config.output_dir, "train.csv")
        test_path = os.path.join(config.output_dir, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        return train_path, test_path


def run_ingestion(config: DataIngestionConfig) -> Tuple[str, str]:
        """
        High-level entrypoint. Reads, validates, preprocesses, splits and saves data.
        Returns tuple of saved train/test paths.
        """
        df = read_input(config.input_path, allowed_extensions=config.allowed_extensions, read_csv_kwargs=config.read_csv_kwargs)
        ok, missing = validate_schema(df, config.required_columns)
        if not ok:
                raise ValueError(f"Input data is missing required columns: {missing}")
        df = preprocess(df, target_column=config.target_column)
        train_path, test_path = split_and_save(df, config)
        return train_path, test_path


if __name__ == "__main__":
        # simple CLI for quick use

        parser = argparse.ArgumentParser(description="Data ingestion for diet recommendation planner")
        parser.add_argument("input_path", help="Input file or directory")
        parser.add_argument("--output_dir", default="data/processed")
        parser.add_argument("--test_size", type=float, default=0.2)
        parser.add_argument("--random_state", type=int, default=42)
        parser.add_argument("--target_column", default=None)
        parser.add_argument("--required_columns", nargs="*", default=None)
        parser.add_argument("--read_csv_kwargs", default=None, help="JSON string of kwargs passed to pandas.read_csv")
        args = parser.parse_args()

        read_kwargs = json.loads(args.read_csv_kwargs) if args.read_csv_kwargs else {}
        cfg = DataIngestionConfig(
                input_path=args.input_path,
                output_dir=args.output_dir,
                test_size=args.test_size,
                random_state=args.random_state,
                target_column=args.target_column,
                required_columns=args.required_columns,
                read_csv_kwargs=read_kwargs,
        )

        train_file, test_file = run_ingestion(cfg)
        print(f"Saved train -> {train_file}")
        print(f"Saved test  -> {test_file}")