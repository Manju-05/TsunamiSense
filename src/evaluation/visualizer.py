"""
Evaluation Visualizer Module
Replicates IEEE paper figures:
- Fig 4: Confusion Matrix for Logistic Regression
- Fig 5: Confusion Matrix for Random Forest
- Fig 6: Confusion Matrix for Support Vector Machine
- Fig 7: ROC curves comparison (LR, RF, SVM)
- Fig 8: Feature importance of the input variables for the RF model
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import AppConfig, load_config
from src.evaluation.metrics import ModelEvaluationReport


class ModelVisualizer:
    """Generates publication-quality figures matching IEEE paper specifications."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.figures_dir = self.config.paths.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="white", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

    def plot_confusion_matrix(self, report: ModelEvaluationReport, filename: str, fig_num: int) -> Path:
        """Plots confusion matrix heatmap matching Fig 4 / 5 / 6 style."""
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        matrix = np.array([[report.tn, report.fp], [report.fn, report.tp]])

        # Paper uses blues colormap
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            linewidths=1.0,
            linecolor="white",
        )
        ax.set_title(f"Confusion Matrix - {report.model_name}", fontweight="bold", pad=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")
        ax.set_xticklabels(["0 (No)", "1 (Tsunami)"])
        ax.set_yticklabels(["0 (No)", "1 (Tsunami)"], rotation=0)
        plt.tight_layout()

        out_path = self.figures_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[SAVED] Fig {fig_num} ({report.model_name} Confusion Matrix) -> {out_path}")
        return out_path

    def plot_roc_curves_comparison(self, reports: List[ModelEvaluationReport]) -> Path:
        """Plots Fig. 7: ROC curves comparison for LR, RF, and SVM."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

        colors = {
            "Logistic Regression": "#1f77b4",
            "Random Forest": "#2ca02c",
            "Support Vector Machine": "#ff7f0e",
        }

        sublabels = ["(a)", "(b)", "(c)"]

        for idx, report in enumerate(reports):
            ax = axes[idx]
            color = colors.get(report.model_name, "#333333")
            ax.plot(
                report.fpr,
                report.tpr,
                color=color,
                lw=2.5,
                label=f"{report.model_name} (AUC = {report.auc_roc:.2f})",
            )
            ax.plot([0, 1], [0, 1], color="#d95f02", linestyle="--", lw=1.5, alpha=0.7)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            if idx == 0:
                ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{sublabels[idx]} {report.model_name}", fontweight="bold")
            ax.legend(loc="lower right", frameon=True)
            ax.grid(True, linestyle=":", alpha=0.6)

        fig.suptitle("ROC Curves Comparison of LR, RF, and SVM (Fig. 7)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_path = self.figures_dir / "fig7_roc_curves_comparison.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SAVED] Fig 7 (Comparative ROC Curves) -> {out_path}")
        return out_path

    def plot_feature_importance(self, importance_dict: Dict[str, float]) -> Path:
        """Plots Fig. 8: Feature importance of the input variables for the RF model."""
        fig, ax = plt.subplots(figsize=(8, 4.5))

        # Sort features by importance ascending for horizontal bar chart
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=False)
        features = [k for k, v in sorted_items]
        values = [v for k, v in sorted_items]

        bars = ax.barh(features, values, color="#3470a3", edgecolor="#1a4163", height=0.55)

        # Annotate percentages on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1%}",
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color="#1a4163",
            )

        ax.set_title("Feature Importance of Input Variables (Random Forest)", fontweight="bold", pad=12)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_xlim(0, max(values) + 0.08)
        ax.grid(axis="x", linestyle=":", alpha=0.7)
        plt.tight_layout()

        out_path = self.figures_dir / "fig8_rf_feature_importance.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[SAVED] Fig 8 (Random Forest Feature Importance) -> {out_path}")
        return out_path
