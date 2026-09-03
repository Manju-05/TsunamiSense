"""
Logistic Regression Model Wrapper
Implements Equation 1 from IEEE Paper:
P(y = 1 | X) = 1 / (1 + e^{-(beta_0 + sum beta_j * x_j)})
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.base import BaseSeismicModel


class TsunamiLogisticRegression(BaseSeismicModel):
    """Logistic Regression baseline classifier for tsunami prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "solver": "lbfgs",
            "max_iter": 1000,
            "C": 1.0,
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        super().__init__(name="Logistic Regression", params=default_params)
        self.build_model()

    def build_model(self) -> None:
        """Instantiates Scikit-Learn LogisticRegression."""
        self.model = LogisticRegression(**self.params)

    @property
    def coefficients(self) -> np.ndarray:
        """Returns feature weights (beta_1, beta_2, beta_3, beta_4)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.model.coef_[0]

    @property
    def intercept(self) -> float:
        """Returns the bias / intercept term (beta_0)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return float(self.model.intercept_[0])
