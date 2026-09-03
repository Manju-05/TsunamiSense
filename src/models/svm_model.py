"""
Support Vector Machine (SVM) Model Wrapper
Implements Equation 4 from IEEE Paper:
f(x) = sign(sum alpha_i * y_i * K(x_i, x) + b)
"""

from typing import Any, Dict, Optional
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.models.base import BaseSeismicModel


class TsunamiSVM(BaseSeismicModel):
    """Kernel Support Vector Classifier for tsunami prediction."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        default_params = {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        super().__init__(name="Support Vector Machine", params=default_params)
        self.build_model()

    def build_model(self) -> None:
        """Instantiates Scikit-Learn SVC with calibrated probabilities."""
        svc_params = {k: v for k, v in self.params.items() if k != "probability"}
        base_svc = SVC(**svc_params)
        try:
            self.model = CalibratedClassifierCV(estimator=base_svc, ensemble=False)
        except TypeError:
            self.model = CalibratedClassifierCV(base_estimator=base_svc)

    @property
    def n_support_vectors(self) -> int:
        """Returns the count of support vectors defining the maximum margin."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if hasattr(self.model, "calibrated_classifiers_"):
            estimator = self.model.calibrated_classifiers_[0].estimator
            if hasattr(estimator, "n_support_"):
                return int(np.sum(estimator.n_support_))
        return 0
