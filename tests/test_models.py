"""
Unit tests for Machine Learning model wrappers and training orchestrator.
"""

from pathlib import Path
import numpy as np
import pytest
from src.config import load_config
from src.models.logistic_regression import TsunamiLogisticRegression
from src.models.svm_model import TsunamiSVM
from src.models.random_forest import TsunamiRandomForest
from src.models.trainer import ModelTrainer
from src.data.fetcher import USGSDataFetcher
from src.preprocessing.pipeline import SeismicPreprocessor


@pytest.fixture
def synthetic_data():
    """Generates small synthetic train/test matrices for fast model testing."""
    np.random.seed(42)
    X_train = np.random.randn(100, 4)
    y_train = np.random.binomial(1, 0.35, size=100)
    X_test = np.random.randn(25, 4)
    return X_train, y_train, X_test


def test_logistic_regression(synthetic_data):
    X_train, y_train, X_test = synthetic_data
    lr = TsunamiLogisticRegression()
    lr.fit(X_train, y_train)

    preds = lr.predict(X_test)
    probas = lr.predict_proba(X_test)

    assert preds.shape == (25,)
    assert set(preds).issubset({0, 1})
    assert probas.shape == (25, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)
    assert len(lr.coefficients) == 4
    assert isinstance(lr.intercept, float)


def test_svm_model(synthetic_data):
    X_train, y_train, X_test = synthetic_data
    svm = TsunamiSVM()
    svm.fit(X_train, y_train)

    preds = svm.predict(X_test)
    probas = svm.predict_proba(X_test)

    assert preds.shape == (25,)
    assert set(preds).issubset({0, 1})
    assert probas.shape == (25, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)
    assert svm.n_support_vectors > 0


def test_random_forest_model(synthetic_data):
    X_train, y_train, X_test = synthetic_data
    rf = TsunamiRandomForest(params={"n_estimators": 20, "max_depth": 5, "min_samples_split": 2, "random_state": 42})
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    probas = rf.predict_proba(X_test)

    assert preds.shape == (25,)
    assert set(preds).issubset({0, 1})
    assert probas.shape == (25, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)

    feature_names = ["magnitude", "depth", "latitude", "longitude"]
    imp_dict = rf.get_feature_importance_dict(feature_names)
    assert len(imp_dict) == 4
    assert np.isclose(sum(imp_dict.values()), 1.0)


def test_model_serialization(tmp_path, synthetic_data):
    X_train, y_train, X_test = synthetic_data
    rf = TsunamiRandomForest(params={"n_estimators": 10, "random_state": 42})
    rf.fit(X_train, y_train)

    save_path = tmp_path / "test_rf.joblib"
    rf.save(save_path)
    assert save_path.exists()

    loaded_rf = TsunamiRandomForest.load(save_path)
    assert loaded_rf.is_fitted is True

    original_preds = rf.predict(X_test)
    loaded_preds = loaded_rf.predict(X_test)
    np.testing.assert_array_equal(original_preds, loaded_preds)
