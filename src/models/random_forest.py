"""
Random Forest Model Wrapper
Implements Equation 2 & 3 from IEEE Paper:
y_hat = majority_vote { h_1(X), h_2(X), ..., h_T(X) }
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseSeismicModel


class TsunamiRandomForest(BaseSeismicModel):
    """Ensemble Random Forest classifier for non-linear tsunami risk prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "n_estimators": 50,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        super().__init__(name="Random Forest", params=default_params)
        self.build_model()

    def build_model(self) -> None:
        """Instantiates Scikit-Learn RandomForestClassifier."""
        self.model = RandomForestClassifier(**self.params)

    @property
    def feature_importances(self) -> np.ndarray:
        """Returns Gini impurity-based feature importances across all trees."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.model.feature_importances_

    def get_feature_importance_dict(self, feature_names: List[str]) -> Dict[str, float]:
        """Maps feature names to their relative importance percentages."""
        importances = self.feature_importances
        return {name: float(imp) for name, imp in zip(feature_names, importances)}
