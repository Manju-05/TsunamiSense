"""
Data Validator Module
Enforces geophysical domain constraints and schema integrity on earthquake records.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np


class SeismicDataValidator:
    """Validates raw seismic datasets against geophysical physical limits and schema rules."""

    REQUIRED_COLUMNS: List[str] = ["magnitude", "depth", "latitude", "longitude", "tsunami"]

    # Geophysical domain boundaries
    BOUNDS = {
        "magnitude": (0.0, 10.0),      # Richter / Moment magnitude
        "depth": (0.0, 800.0),          # Hypocenter depth in km
        "latitude": (-90.0, 90.0),      # Geographic latitude
        "longitude": (-180.0, 180.0),   # Geographic longitude
        "tsunami": (0, 1),              # Binary flag
    }

    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Checks if all required columns exist in the DataFrame."""
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            return False, [f"Missing required columns: {missing}"]
        return True, []

    @classmethod
    def validate_ranges(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Checks if numeric values fall within valid physical limits."""
        issues = []
        for col, (min_val, max_val) in cls.BOUNDS.items():
            if col in df.columns:
                invalid_rows = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(invalid_rows) > 0:
                    issues.append(
                        f"Column '{col}' has {len(invalid_rows)} values outside valid bounds [{min_val}, {max_val}]."
                    )

        return len(issues) == 0, issues

    @classmethod
    def sanitize(cls, df: pd.DataFrame, min_magnitude: float = 6.0) -> pd.DataFrame:
        """
        Cleans and filters the DataFrame:
        - Drops rows with corrupted non-numeric features
        - Filters magnitude >= min_magnitude as required by the study
        - Ensures binary target (0 or 1)
        """
        df_clean = df.copy()
        for col in ["magnitude", "depth", "latitude", "longitude"]:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        if "tsunami" in df_clean.columns:
            df_clean["tsunami"] = pd.to_numeric(df_clean["tsunami"], errors="coerce").fillna(0).astype(int)
            df_clean["tsunami"] = df_clean["tsunami"].apply(lambda v: 1 if v > 0 else 0)

        # Filter minimum magnitude threshold
        df_clean = df_clean[df_clean["magnitude"] >= min_magnitude]
        return df_clean.reset_index(drop=True)
