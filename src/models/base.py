"""
Base Model Abstract Interface
Defines unified contract for all tsunami classification models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np


class BaseSeismicModel(ABC):
    """Abstract Base Class for all earthquake-tsunami classifiers."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.model: Any = None
        self.is_fitted: bool = False

    @abstractmethod
    def build_model(self) -> None:
        """Instantiates the underlying scikit-learn estimator."""
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseSeismicModel":
        """Fits the classifier on standardized training features."""
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary tsunami labels (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling predict().")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities [P(y=0), P(y=1)]."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling predict_proba().")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            # Fallback for margin-based models without probability calibration
            decision = self.model.decision_function(X)
            proba_pos = 1.0 / (1.0 + np.exp(-decision))
            return np.column_stack([1.0 - proba_pos, proba_pos])
        else:
            raise NotImplementedError(f"Model '{self.name}' does not support probability estimation.")

    def save(self, filepath: Path) -> Path:
        """Serializes the fitted model artifact to disk."""
        if not self.is_fitted:
            raise RuntimeError(f"Cannot save unfitted model '{self.name}'.")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "BaseSeismicModel":
        """Deserializes a fitted model artifact from disk."""
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at: {filepath}")
        loaded = joblib.load(filepath)
        return loaded
