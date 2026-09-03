"""
Comprehensive Evaluation Benchmark Runner
Executes testing across LR, SVM, and RF models, generates Table 1 metrics,
plots Figures 4, 5, 6, 7, and 8, and tests the Samoa 2009 Case Study.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.config import AppConfig, load_config
from src.data.fetcher import USGSDataFetcher
from src.preprocessing.pipeline import SeismicPreprocessor
from src.models.logistic_regression import TsunamiLogisticRegression
from src.models.svm_model import TsunamiSVM
from src.models.random_forest import TsunamiRandomForest
from src.evaluation.metrics import ModelEvaluationReport, SeismicMetricsCalculator
from src.evaluation.visualizer import ModelVisualizer
from src.inference.predict import TsunamiInferenceService


class SeismicBenchmarkRunner:
    """Orchestrates comprehensive model evaluation against IEEE paper standards."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.visualizer = ModelVisualizer(self.config)

    def run_benchmark(self) -> Tuple[List[ModelEvaluationReport], Dict[str, any]]:
        """Executes full benchmark evaluation."""
        print("\n" + "=" * 70)
        print("  AI-DRIVEN TSUNAMI PREDICTION: BENCHMARK & EVALUATION SUITE")
        print("=" * 70)

        # 1. Load data and run leak-free preprocessing
        fetcher = USGSDataFetcher(self.config)
        df = fetcher.get_dataset(force_refresh=False)
        preprocessor = SeismicPreprocessor(self.config)
        dataset = preprocessor.fit_transform_split(df)

        X_test = dataset.X_test
        y_test = dataset.y_test

        # 2. Load trained model artifacts
        models_dir = self.config.paths.models_dir
        lr_model = TsunamiLogisticRegression.load(models_dir / "logistic_regression.joblib")
        svm_model = TsunamiSVM.load(models_dir / "support_vector_machine.joblib")
        rf_model = TsunamiRandomForest.load(models_dir / "random_forest.joblib")

        models = [lr_model, svm_model, rf_model]
        reports: List[ModelEvaluationReport] = []

        print("\n[EVALUATION] Scoring models on unseen test dataset (N = %d)..." % len(y_test))

        for model in models:
            preds = model.predict(X_test)
            probas = model.predict_proba(X_test)[:, 1]
            report = SeismicMetricsCalculator.evaluate_model(
                model_name=model.name,
                y_true=y_test,
                y_pred=preds,
                y_prob=probas
            )
            reports.append(report)

        # 3. Print Quantitative Table 1 Comparison
        print("\n" + "-" * 85)
        print(f"{'Classifier':<26} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'F2':<6} | {'AUC':<6} | {'PR-AUC':<6} | {'MCC':<6}")
        print("-" * 85)
        for r in reports:
            print(f"{r.model_name:<26} | {r.accuracy:<8.3f} | {r.precision:<9.3f} | {r.recall:<8.3f} | {r.f1_score:<6.3f} | {r.f2_score:<6.3f} | {r.auc_roc:<6.3f} | {r.pr_auc:<6.3f} | {r.mcc:<6.3f}")
        print("-" * 85)

        # 4. Generate IEEE Paper Figures
        print("\n[VISUALIZATION] Generating evaluation figures matching research paper...")
        # Fig 4: Confusion Matrix for LR
        lr_rep = next(r for r in reports if r.model_name == "Logistic Regression")
        self.visualizer.plot_confusion_matrix(lr_rep, "fig4_lr_confusion_matrix.png", fig_num=4)

        # Fig 5: Confusion Matrix for RF
        rf_rep = next(r for r in reports if r.model_name == "Random Forest")
        self.visualizer.plot_confusion_matrix(rf_rep, "fig5_rf_confusion_matrix.png", fig_num=5)

        # Fig 6: Confusion Matrix for SVM
        svm_rep = next(r for r in reports if r.model_name == "Support Vector Machine")
        self.visualizer.plot_confusion_matrix(svm_rep, "fig6_svm_confusion_matrix.png", fig_num=6)

        # Fig 7: Comparative ROC Curves
        self.visualizer.plot_roc_curves_comparison([lr_rep, rf_rep, svm_rep])

        # Fig 8: Feature Importance for RF
        importances = rf_model.get_feature_importance_dict(self.config.data.features)
        self.visualizer.plot_feature_importance(importances)

        # 5. Export JSON summary
        metrics_file = self.config.paths.reports_dir / "metrics_summary.json"
        SeismicMetricsCalculator.export_summary(reports, metrics_file)

        # 6. Run Historical Case Study Validation (2009 Samoa Mw 8.1)
        inference_service = TsunamiInferenceService(self.config)
        case_study_res = inference_service.validate_samoa_case_study()

        return reports, case_study_res


if __name__ == "__main__":
    runner = SeismicBenchmarkRunner()
    runner.run_benchmark()
