"""
Unit tests for scientific metrics calculations (F2, MCC, PR-AUC, ROC-AUC).
"""

import numpy as np
import pytest
from src.evaluation.metrics import SeismicMetricsCalculator


def test_f2_score_calculation():
    # If precision=0.8 and recall=0.8, F2 should equal 0.8
    f2 = SeismicMetricsCalculator.compute_f2_score(0.8, 0.8)
    assert np.isclose(f2, 0.8)

    # When recall (1.0) is higher than precision (0.5), F2 should be closer to recall
    f2_weighted = SeismicMetricsCalculator.compute_f2_score(0.5, 1.0)
    # F2 = 5 * (0.5 * 1.0) / (4*0.5 + 1.0) = 2.5 / 3.0 = 0.8333...
    assert np.isclose(f2_weighted, 5 / 6)


def test_mcc_calculation():
    # Perfect confusion matrix: TP=50, TN=50, FP=0, FN=0 -> MCC should be 1.0
    mcc_perf = SeismicMetricsCalculator.compute_mcc(tp=50, tn=50, fp=0, fn=0)
    assert np.isclose(mcc_perf, 1.0)

    # Completely wrong: TP=0, TN=0, FP=50, FN=50 -> MCC should be -1.0
    mcc_inv = SeismicMetricsCalculator.compute_mcc(tp=0, tn=0, fp=50, fn=50)
    assert np.isclose(mcc_inv, -1.0)


def test_evaluate_model_pipeline():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.1, 0.85, 0.95, 0.15])

    report = SeismicMetricsCalculator.evaluate_model(
        model_name="TestClassifier",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )

    assert report.accuracy == 0.9
    assert report.tp == 4
    assert report.fn == 1
    assert report.tn == 5
    assert report.fp == 0
    assert report.auc_roc > 0.9
    assert report.f2_score > 0.8
