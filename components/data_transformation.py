from components.data_transformation import DataTransformation, DataTransformationConfig
from dataclasses import dataclass
import os
import pickle
from typing import List, Tuple, Union
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

"""
components/data_transformation.py

Data transformation utilities for preprocessing datasets prior to model training.
Provides a configurable sklearn-based transformer builder and helpers to apply and
persist the preprocessing pipeline.

Usage:
    config = DataTransformationConfig(preprocessor_obj_file_path="artifacts/preprocessor.pkl")
    dt = DataTransformation(config)
    train_arr, test_arr, preprocessor_path = dt.initiate_data_transformation(
        train_df=train_dataframe,
        test_df=test_dataframe,
        target_column="target",
        numeric_features=[...],
        categorical_features=[...]
    )
"""




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = "artifacts/preprocessor.pkl"


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    def _read_df(self, df_or_path: Union[pd.DataFrame, str]) -> pd.DataFrame:
        if isinstance(df_or_path, pd.DataFrame):
            return df_or_path.copy()
        if isinstance(df_or_path, str):
            if not os.path.exists(df_or_path):
                raise FileNotFoundError(f"File not found: {df_or_path}")
            return pd.read_csv(df_or_path)
        raise TypeError("train_df/test_df must be a pandas DataFrame or a CSV file path string")

    def get_preprocessor_object(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        numeric_imputer_strategy: str = "median",
        categorical_imputer_strategy: str = "most_frequent",
    ) -> ColumnTransformer:
        """
        Build and return a ColumnTransformer that imputes and scales numeric features
        and imputes and one-hot encodes categorical features.
        """
        # Numeric pipeline: impute -> scale
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=numeric_imputer_strategy)),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical pipeline: impute -> one-hot encode
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_imputer_strategy, fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

        return preprocessor

    def _save_object(self, file_path: str, obj) -> None:
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    def _load_object(self, file_path: str):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def initiate_data_transformation(
        self,
        train_df: Union[pd.DataFrame, str],
        test_df: Union[pd.DataFrame, str],
        target_column: str,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> Tuple:
        """
        Fit the preprocessor on train_df and transform both train_df and test_df.

        Returns:
            train_arr: numpy array of transformed features concatenated with target as last column
            test_arr: numpy array of transformed features concatenated with target as last column
            preprocessor_path: path where the preprocessor object is saved
        """
        train_df = self._read_df(train_df)
        test_df = self._read_df(test_df)

        if target_column not in train_df.columns:
            raise KeyError(f"Target column '{target_column}' not found in train_df")
        if target_column not in test_df.columns:
            raise KeyError(f"Target column '{target_column}' not found in test_df")

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column].values
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column].values

        preprocessor = self.get_preprocessor_object(numeric_features, categorical_features)

        # Fit on training features
        preprocessor.fit(X_train)

        # Transform both train and test
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Concatenate features and target as arrays

        train_arr = np.c_[X_train_transformed, y_train]
        test_arr = np.c_[X_test_transformed, y_test]

        # Persist the preprocessor
        self._save_object(self.config.preprocessor_obj_file_path, preprocessor)

        return train_arr, test_arr, self.config.preprocessor_obj_file_path