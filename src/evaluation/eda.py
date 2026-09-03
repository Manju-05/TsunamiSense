"""
Exploratory Data Analysis (EDA) Module
Replicates IEEE paper figures:
- Fig 1: Magnitude distribution histogram & density curve
- Fig 2: Global geographic distribution of earthquakes (scatter map)
- Fig 3: Depth distribution histogram & density curve
- Correlation Heatmap
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import AppConfig, load_config


class SeismicEDA:
    """Generates publication-quality visualizations matching the IEEE paper."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.figures_dir = self.config.paths.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Apply standard clean styling
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14,
        })

    def plot_magnitude_distribution(self, df: pd.DataFrame) -> Path:
        """Replicates Fig. 1: Distribution of earthquake magnitudes."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            df["magnitude"],
            kde=True,
            color="#2b7bba",
            edgecolor="#1c4e75",
            bins=20,
            ax=ax,
            line_kws={"linewidth": 2, "color": "#0d2b45"}
        )
        ax.set_title("Distribution of Earthquake Magnitudes (USGS 2015–2025)", fontweight="bold", pad=12)
        ax.set_xlabel("Magnitude (Richter / Mw)")
        ax.set_ylabel("Count")
        ax.set_xlim(5.9, 8.5)
        plt.tight_layout()

        out_path = self.figures_dir / "fig1_magnitude_distribution.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[SAVED] Fig 1 (Magnitude distribution) -> {out_path}")
        return out_path

    def plot_global_distribution(self, df: pd.DataFrame) -> Path:
        """Replicates Fig. 2: Global distribution of earthquakes with magnitudes ranging from 6.0 to 8.0."""
        fig, ax = plt.subplots(figsize=(10, 5.5))
        scatter = ax.scatter(
            df["longitude"],
            df["latitude"],
            c=df["magnitude"],
            cmap="magma_r",
            s=(df["magnitude"] - 5.5) ** 3 * 6,
            alpha=0.65,
            edgecolors="black",
            linewidth=0.4,
        )
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Magnitude (Mw)", rotation=270, labelpad=15)

        ax.set_title("Global Geographic Distribution of Earthquakes (M >= 6.0)", fontweight="bold", pad=12)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        plt.tight_layout()

        out_path = self.figures_dir / "fig2_global_distribution.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[SAVED] Fig 2 (Global map distribution) -> {out_path}")
        return out_path

    def plot_depth_distribution(self, df: pd.DataFrame) -> Path:
        """Replicates Fig. 3: Depth distribution of earthquakes."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(
            df["depth"],
            kde=True,
            color="#20857c",
            edgecolor="#114a44",
            bins=25,
            ax=ax,
            line_kws={"linewidth": 2, "color": "#072421"}
        )
        ax.set_title("Depth Distribution of Earthquakes (Hypocenter Depth in km)", fontweight="bold", pad=12)
        ax.set_xlabel("Depth (km)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 700)
        plt.tight_layout()

        out_path = self.figures_dir / "fig3_depth_distribution.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[SAVED] Fig 3 (Depth distribution) -> {out_path}")
        return out_path

    def plot_feature_correlations(self, df: pd.DataFrame) -> Path:
        """Generates correlation heatmap across seismic features and the tsunami target."""
        fig, ax = plt.subplots(figsize=(7, 5.5))
        cols = ["magnitude", "depth", "latitude", "longitude", "tsunami"]
        corr = df[cols].corr()

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.8,
            square=True,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title("Correlation Matrix (Seismic Features vs Tsunami Target)", fontweight="bold", pad=12)
        plt.tight_layout()

        out_path = self.figures_dir / "feature_correlations.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[SAVED] Feature Correlations -> {out_path}")
        return out_path

    def run_all_eda(self, df: pd.DataFrame):
        """Executes full exploratory analysis suite."""
        print("[EDA] Running Exploratory Data Analysis & generating IEEE paper figures...")
        self.plot_magnitude_distribution(df)
        self.plot_global_distribution(df)
        self.plot_depth_distribution(df)
        self.plot_feature_correlations(df)
        print("[DONE] All EDA visualizations successfully generated.")
