"""
Evaluation Metrics Module
Implements comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1-Score, F2-Score (Eq. 9)
- Matthews Correlation Coefficient (MCC, Eq. 10)
- ROC-AUC, PR-AUC, and Confusion Matrix breakdown
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)


@dataclass
class ModelEvaluationReport:
    """Holds all quantitative metrics and confusion matrix elements for a model."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f2_score: float
    auc_roc: float
    pr_auc: float
    mcc: float
    tp: int
    tn: int
    fp: int
    fn: int
    fpr: List[float]
    tpr: List[float]
    precision_curve: List[float]
    recall_curve: List[float]

    def to_dict(self, include_curves: bool = False) -> Dict[str, Any]:
        """Converts evaluation metrics to dictionary format."""
        data = asdict(self)
        if not include_curves:
            data.pop("fpr", None)
            data.pop("tpr", None)
            data.pop("precision_curve", None)
            data.pop("recall_curve", None)
        return data


class SeismicMetricsCalculator:
    """Calculates all scientific evaluation metrics defined in the IEEE paper."""

    @staticmethod
    def compute_f2_score(precision: float, recall: float) -> float:
        """
        Computes F2-Score using Equation 9:
        F2 = (1 + 2^2) * (Precision * Recall) / ((2^2 * Precision) + Recall)
        """
        if (4 * precision + recall) == 0:
            return 0.0
        return (5.0 * precision * recall) / (4.0 * precision + recall)

    @staticmethod
    def compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Computes Matthews Correlation Coefficient using Equation 10:
        MCC = (TP * TN - FP * FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
        """
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    @classmethod
    def evaluate_model(
        cls,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> ModelEvaluationReport:
        """Computes all evaluation metrics from true labels, predictions, and predicted probabilities."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f2 = cls.compute_f2_score(prec, rec)
        mcc = cls.compute_mcc(tp, tn, fp, fn)

        # ROC Curve & AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_roc = float(roc_auc_score(y_true, y_prob))

        # Precision-Recall Curve & PR-AUC
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))

        return ModelEvaluationReport(
            model_name=model_name,
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            f2_score=round(f2, 4),
            auc_roc=round(auc_roc, 4),
            pr_auc=round(pr_auc, 4),
            mcc=round(mcc, 4),
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            precision_curve=p_curve.tolist(),
            recall_curve=r_curve.tolist(),
        )

    @classmethod
    def export_summary(cls, reports: List[ModelEvaluationReport], output_path: Path) -> None:
        """Exports metrics summary to JSON for reporting."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {rep.model_name: rep.to_dict(include_curves=False) for rep in reports}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVED] Exported metrics summary -> {output_path}")
