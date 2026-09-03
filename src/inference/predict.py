"""
Inference & Early Warning Operational Scoring Module
Loads production artifacts and delivers real-time tsunami risk probabilities.
Includes validation against historical benchmark events (e.g. 2009 Samoa Mw 8.1).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from src.config import AppConfig, load_config
from src.models.logistic_regression import TsunamiLogisticRegression
from src.models.svm_model import TsunamiSVM
from src.models.random_forest import TsunamiRandomForest
from src.preprocessing.pipeline import SeismicPreprocessor


@dataclass
class TsunamiRiskPrediction:
    """Encapsulates the consensus early warning risk assessment across all classifiers."""
    magnitude: float
    depth: float
    latitude: float
    longitude: float
    lr_probability: float
    lr_prediction: int
    svm_probability: float
    svm_prediction: int
    rf_probability: float
    rf_prediction: int
    consensus_risk: str
    risk_level: str


class TsunamiInferenceService:
    """Production inference service for rapid earthquake classification."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.models_dir = self.config.paths.models_dir
        self.preprocessor = SeismicPreprocessor(self.config)
        self.preprocessor.load_artifacts()
        self._load_models()

    def _load_models(self) -> None:
        """Loads serialized models from disk."""
        self.lr_model = TsunamiLogisticRegression.load(self.models_dir / "logistic_regression.joblib")
        self.svm_model = TsunamiSVM.load(self.models_dir / "support_vector_machine.joblib")
        self.rf_model = TsunamiRandomForest.load(self.models_dir / "random_forest.joblib")

    def predict_event(
        self,
        magnitude: float,
        depth: float,
        latitude: float,
        longitude: float
    ) -> TsunamiRiskPrediction:
        """
        Scores a single earthquake event across all three models.
        """
        # 1. Transform raw seismic inputs with saved imputer & standard scaler
        X_scaled = self.preprocessor.transform_single_event(magnitude, depth, latitude, longitude)

        # 2. Score with Logistic Regression
        lr_pred = int(self.lr_model.predict(X_scaled)[0])
        lr_prob = float(self.lr_model.predict_proba(X_scaled)[0, 1])

        # 3. Score with Support Vector Machine
        svm_pred = int(self.svm_model.predict(X_scaled)[0])
        svm_prob = float(self.svm_model.predict_proba(X_scaled)[0, 1])

        # 4. Score with Random Forest
        rf_pred = int(self.rf_model.predict(X_scaled)[0])
        rf_prob = float(self.rf_model.predict_proba(X_scaled)[0, 1])

        # 5. Determine Consensus Early Warning Tier
        max_prob = max(lr_prob, svm_prob, rf_prob)
        votes = lr_pred + svm_pred + rf_pred

        if votes >= 2 and max_prob >= 0.70:
            consensus = "HIGH TSUNAMI RISK - WARNING ISSUED"
            risk_level = "CRITICAL"
        elif votes >= 1 or max_prob >= 0.40:
            consensus = "MODERATE TSUNAMI RISK - ADVISORY"
            risk_level = "ELEVATED"
        else:
            consensus = "LOW TSUNAMI RISK - NO WARNING"
            risk_level = "NORMAL"

        return TsunamiRiskPrediction(
            magnitude=magnitude,
            depth=depth,
            latitude=latitude,
            longitude=longitude,
            lr_probability=round(lr_prob, 4),
            lr_prediction=lr_pred,
            svm_probability=round(svm_prob, 4),
            svm_prediction=svm_pred,
            rf_probability=round(rf_prob, 4),
            rf_prediction=rf_pred,
            consensus_risk=consensus,
            risk_level=risk_level,
        )

    def validate_samoa_case_study(self) -> Dict[str, Any]:
        """
        Replicates Section IV-F: Validation on representative tsunamigenic event (2009 Samoa Mw 8.1).
        """
        cs = self.config.case_study
        print(f"\n[CASE STUDY] Evaluating {cs.name} (Mw {cs.magnitude}, Depth {cs.depth} km, Lat {cs.latitude}, Lon {cs.longitude})...")

        pred = self.predict_event(
            magnitude=cs.magnitude,
            depth=cs.depth,
            latitude=cs.latitude,
            longitude=cs.longitude
        )

        results = {
            "event_name": cs.name,
            "magnitude": cs.magnitude,
            "depth_km": cs.depth,
            "latitude": cs.latitude,
            "longitude": cs.longitude,
            "actual_tsunami": cs.actual_tsunami,
            "logistic_regression_prob": f"{pred.lr_probability * 100:.2f}%",
            "support_vector_machine_prob": f"{pred.svm_probability * 100:.2f}%",
            "random_forest_prob": f"{pred.rf_probability * 100:.2f}%",
            "consensus": pred.consensus_risk,
            "all_detected": (pred.lr_prediction == 1 and pred.svm_prediction == 1 and pred.rf_prediction == 1),
        }

        print("=" * 65)
        print(f"  CASE STUDY VALIDATION: {cs.name}")
        print("=" * 65)
        print(f"  Actual Outcome: {'TSUNAMI GENERATED (1)' if cs.actual_tsunami == 1 else 'NO TSUNAMI (0)'}")
        print(f"  Logistic Regression Probability : {results['logistic_regression_prob']}")
        print(f"  Support Vector Machine Prob    : {results['support_vector_machine_prob']}")
        print(f"  Random Forest Probability      : {results['random_forest_prob']}")
        print(f"  System Consensus Assessment    : {results['consensus']}")
        print("=" * 65)

        return results
