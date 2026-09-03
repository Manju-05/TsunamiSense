"""
Preprocessing Pipeline Module
Implements leak-free median imputation, 80/20 stratified splitting, and feature standardization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig, load_config


@dataclass
class ProcessedDataset:
    """Encapsulates prepared feature matrices, target vectors, and fitted transformation objects."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    imputer: SimpleImputer
    scaler: StandardScaler
    df_raw: pd.DataFrame


class SeismicPreprocessor:
    """Preprocesses raw seismic data following IEEE study guidelines."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.features = self.config.data.features
        self.target = self.config.data.target
        self.test_size = self.config.data.test_size
        self.random_state = self.config.project.random_seed
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_transform_split(self, df: pd.DataFrame) -> ProcessedDataset:
        """
        Executes the full leak-free preprocessing pipeline:
        1. Feature/Target extraction
        2. 80/20 Stratified Train-Test Split
        3. Median Imputation fitted ONLY on X_train
        4. Standard Scaling (z = (x - mu) / sigma) fitted ONLY on X_train
        """
        X = df[self.features].copy()
        y = df[self.target].values.astype(int)

        # 1. Stratified 80/20 Train-Test Split
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.config.data.stratify else None,
            shuffle=self.config.data.shuffle,
        )

        print(f"[SPLIT] Dataset split: Train={len(X_train_df)} ({y_train.sum()} pos), Test={len(X_test_df)} ({y_test.sum()} pos)")

        # 2. Imputation (fit on train, transform both)
        X_train_imputed = self.imputer.fit_transform(X_train_df)
        X_test_imputed = self.imputer.transform(X_test_df)

        # 3. Standardization (fit on train, transform both)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)

        # 4. Save artifacts for production inference
        self.save_artifacts()

        return ProcessedDataset(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.features,
            imputer=self.imputer,
            scaler=self.scaler,
            df_raw=df,
        )

    def transform_single_event(self, magnitude: float, depth: float, latitude: float, longitude: float) -> np.ndarray:
        """Transforms a single live earthquake event for production scoring."""
        raw_df = pd.DataFrame(
            [[magnitude, depth, latitude, longitude]],
            columns=self.features
        )
        imputed = self.imputer.transform(raw_df)
        scaled = self.scaler.transform(imputed)
        return scaled

    def save_artifacts(self) -> None:
        """Serializes fitted imputer and scaler to the processed data directory."""
        proc_dir = self.config.paths.processed_data_dir
        proc_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.imputer, proc_dir / "median_imputer.joblib")
        joblib.dump(self.scaler, proc_dir / "standard_scaler.joblib")
        print(f"[SAVED] Saved preprocessor artifacts (imputer, scaler) to {proc_dir}")

    def load_artifacts(self) -> None:
        """Loads fitted imputer and scaler from disk."""
        proc_dir = self.config.paths.processed_data_dir
        self.imputer = joblib.load(proc_dir / "median_imputer.joblib")
        self.scaler = joblib.load(proc_dir / "standard_scaler.joblib")
