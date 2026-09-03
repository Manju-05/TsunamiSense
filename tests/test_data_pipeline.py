"""
Unit tests for data ingestion, schema validation, and preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.data.validator import SeismicDataValidator
from src.data.fetcher import USGSDataFetcher
from src.preprocessing.pipeline import SeismicPreprocessor


def test_seismic_validator():
    valid_data = pd.DataFrame({
        "magnitude": [6.5, 7.2],
        "depth": [25.0, 110.0],
        "latitude": [-12.5, 35.8],
        "longitude": [140.2, -75.4],
        "tsunami": [1, 0]
    })
    is_valid_schema, _ = SeismicDataValidator.validate_schema(valid_data)
    assert is_valid_schema is True

    is_valid_range, _ = SeismicDataValidator.validate_ranges(valid_data)
    assert is_valid_range is True


def test_data_fetcher_and_distribution():
    config = load_config()
    fetcher = USGSDataFetcher(config)
    df = fetcher.get_dataset(force_refresh=False)

    assert len(df) >= 1300
    assert "tsunami" in df.columns
    assert (df["magnitude"] >= 6.0).all()
    assert (df["depth"] >= 0).all()

    # Check positive / negative classes exist
    counts = df["tsunami"].value_counts().to_dict()
    assert counts.get(1, 0) > 350
    assert counts.get(0, 0) > 800


def test_preprocessing_pipeline_no_leakage():
    config = load_config()
    fetcher = USGSDataFetcher(config)
    df = fetcher.get_dataset(force_refresh=False)

    preprocessor = SeismicPreprocessor(config)
    dataset = preprocessor.fit_transform_split(df)

    # 80/20 train/test split verification
    total_samples = len(df)
    expected_test_len = int(round(total_samples * config.data.test_size))
    expected_train_len = total_samples - expected_test_len

    assert dataset.X_train.shape[1] == 4
    assert dataset.X_test.shape[1] == 4
    assert dataset.X_train.shape[0] == expected_train_len
    assert dataset.X_test.shape[0] == expected_test_len
    assert dataset.y_train.shape[0] == expected_train_len
    assert dataset.y_test.shape[0] == expected_test_len

    # Verify no NaN values
    assert not np.isnan(dataset.X_train).any()
    assert not np.isnan(dataset.X_test).any()

    # Verify scaling properties on training data (mean ~ 0, std ~ 1)
    np.testing.assert_allclose(dataset.X_train.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(dataset.X_train.std(axis=0), 1.0, atol=1e-7)
