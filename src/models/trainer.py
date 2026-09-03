"""
Model Trainer & Hyperparameter Tuning Orchestrator
Coordinates training of Logistic Regression, SVM, and Random Forest models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.config import AppConfig, load_config
from src.models.base import BaseSeismicModel
from src.models.logistic_regression import TsunamiLogisticRegression
from src.models.svm_model import TsunamiSVM
from src.models.random_forest import TsunamiRandomForest
from src.preprocessing.pipeline import ProcessedDataset


@dataclass
class TrainedModelRegistry:
    """Registry holding all trained and fitted model instances."""
    lr: TsunamiLogisticRegression
    svm: TsunamiSVM
    rf: TsunamiRandomForest
    rf_best_params: Dict[str, any]


class ModelTrainer:
    """Trains, tunes, and serializes all models described in the IEEE study."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.models_dir = self.config.paths.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run_rf_grid_search(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, any]:
        """
        Executes cross-validated Grid Search on Random Forest
        to identify optimal tree number, depth, and split thresholds.
        """
        print("[TUNING] Running 5-Fold Grid Search CV for Random Forest...")
        base_rf = TsunamiRandomForest().model
        grid_config = self.config.models.rf_grid

        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=grid_config["param_grid"],
            cv=grid_config["cv"],
            scoring=grid_config["scoring"],
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print(f"[TUNING] Best Random Forest Parameters: {best_params} (Best CV F1: {grid_search.best_score_:.4f})")
        return best_params

    def train_all_models(self, dataset: ProcessedDataset, tune_rf: bool = True) -> TrainedModelRegistry:
        """
        Trains Logistic Regression, SVM, and tuned Random Forest classifiers.
        """
        X_train, y_train = dataset.X_train, dataset.y_train

        # 1. Train Logistic Regression Baseline
        print("[TRAINING] Fitting Logistic Regression...")
        lr_model = TsunamiLogisticRegression(params=self.config.models.lr_params)
        lr_model.fit(X_train, y_train)

        # 2. Train Support Vector Machine
        print("[TRAINING] Fitting Support Vector Machine (RBF Kernel)...")
        svm_model = TsunamiSVM(params=self.config.models.svm_params)
        svm_model.fit(X_train, y_train)

        # 3. Optimize and Train Random Forest
        if tune_rf:
            best_rf_params = self.run_rf_grid_search(X_train, y_train)
            rf_model = TsunamiRandomForest(params=best_rf_params)
        else:
            rf_model = TsunamiRandomForest(params=self.config.models.rf_params)
            best_rf_params = self.config.models.rf_params

        print("[TRAINING] Fitting Optimized Random Forest...")
        rf_model.fit(X_train, y_train)

        # 4. Save model artifacts
        self.save_models(lr_model, svm_model, rf_model)

        print("[SUCCESS] All three models trained and serialized successfully.")
        return TrainedModelRegistry(
            lr=lr_model,
            svm=svm_model,
            rf=rf_model,
            rf_best_params=best_rf_params,
        )

    def save_models(self, lr: TsunamiLogisticRegression, svm: TsunamiSVM, rf: TsunamiRandomForest) -> None:
        """Persists trained model objects to models/saved_models/."""
        lr.save(self.models_dir / "logistic_regression.joblib")
        svm.save(self.models_dir / "support_vector_machine.joblib")
        rf.save(self.models_dir / "random_forest.joblib")
        print(f"[SAVED] Serialized all models to: {self.models_dir}")
