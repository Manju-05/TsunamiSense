"""
Configuration Loader Module
Parses config/config.yaml into structured, strongly-typed settings objects.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    random_seed: int


@dataclass(frozen=True)
class PathConfig:
    raw_data_dir: Path
    processed_data_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path
    raw_dataset_file: Path


@dataclass(frozen=True)
class DataConfig:
    min_magnitude: float
    start_year: int
    end_year: int
    features: List[str]
    target: str
    test_size: float
    shuffle: bool
    stratify: bool


@dataclass(frozen=True)
class ModelParamsConfig:
    lr_params: Dict[str, Any]
    svm_params: Dict[str, Any]
    rf_params: Dict[str, Any]
    rf_grid: Dict[str, Any]


@dataclass(frozen=True)
class CaseStudyConfig:
    name: str
    date: str
    magnitude: float
    depth: float
    latitude: float
    longitude: float
    actual_tsunami: int


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    paths: PathConfig
    data: DataConfig
    models: ModelParamsConfig
    case_study: CaseStudyConfig


def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Loads and validates the YAML configuration file."""
    root = get_project_root()
    if config_path is None:
        config_path = root / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve paths relative to project root
    paths = PathConfig(
        raw_data_dir=root / cfg["paths"]["raw_data_dir"],
        processed_data_dir=root / cfg["paths"]["processed_data_dir"],
        models_dir=root / cfg["paths"]["models_dir"],
        reports_dir=root / cfg["paths"]["reports_dir"],
        figures_dir=root / cfg["paths"]["figures_dir"],
        raw_dataset_file=root / cfg["paths"]["raw_dataset_file"],
    )

    # Ensure output directories exist
    paths.raw_data_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    project = ProjectConfig(
        name=cfg["project"]["name"],
        version=cfg["project"]["version"],
        random_seed=cfg["project"]["random_seed"],
    )

    data = DataConfig(
        min_magnitude=cfg["data"]["min_magnitude"],
        start_year=cfg["data"]["start_year"],
        end_year=cfg["data"]["end_year"],
        features=cfg["data"]["features"],
        target=cfg["data"]["target"],
        test_size=cfg["data"]["test_size"],
        shuffle=cfg["data"]["shuffle"],
        stratify=cfg["data"]["stratify"],
    )

    models = ModelParamsConfig(
        lr_params=cfg["models"]["logistic_regression"],
        svm_params=cfg["models"]["svm"],
        rf_params=cfg["models"]["random_forest"],
        rf_grid=cfg["models"]["grid_search_rf"],
    )

    case_study = CaseStudyConfig(
        name=cfg["case_study"]["name"],
        date=cfg["case_study"]["date"],
        magnitude=cfg["case_study"]["magnitude"],
        depth=cfg["case_study"]["depth"],
        latitude=cfg["case_study"]["latitude"],
        longitude=cfg["case_study"]["longitude"],
        actual_tsunami=cfg["case_study"]["actual_tsunami"],
    )

    return AppConfig(
        project=project,
        paths=paths,
        data=data,
        models=models,
        case_study=case_study,
    )
