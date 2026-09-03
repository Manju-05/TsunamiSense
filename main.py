"""
AI-Driven Tsunami Prediction - Master CLI Entrypoint
Provides commands for data harvesting, EDA, model training, evaluation, inference, and web serving.
"""

import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.data.fetcher import USGSDataFetcher
from src.preprocessing.pipeline import SeismicPreprocessor
from src.evaluation.eda import SeismicEDA
from src.models.trainer import ModelTrainer
from src.evaluation.benchmark import SeismicBenchmarkRunner
from src.inference.predict import TsunamiInferenceService


def cmd_data(args):
    """Harvests USGS data and validates preprocessing pipeline."""
    cfg = load_config()
    fetcher = USGSDataFetcher(cfg)
    df = fetcher.get_dataset(force_refresh=args.refresh)
    prep = SeismicPreprocessor(cfg)
    dataset = prep.fit_transform_split(df)
    print(f"[SUCCESS] Dataset preprocessed: {dataset.X_train.shape[0]} train rows, {dataset.X_test.shape[0]} test rows.")


def cmd_eda(args):
    """Generates all exploratory data analysis figures."""
    cfg = load_config()
    fetcher = USGSDataFetcher(cfg)
    df = fetcher.get_dataset(force_refresh=False)
    eda = SeismicEDA(cfg)
    eda.run_all_eda(df)
    print(f"[SUCCESS] EDA figures generated in: {cfg.paths.figures_dir}")


def cmd_train(args):
    """Trains LR, SVM, and hyperparameter-tuned RF models."""
    cfg = load_config()
    fetcher = USGSDataFetcher(cfg)
    df = fetcher.get_dataset(force_refresh=False)
    prep = SeismicPreprocessor(cfg)
    dataset = prep.fit_transform_split(df)
    trainer = ModelTrainer(cfg)
    trainer.train_all_models(dataset, tune_rf=not args.no_tune)
    print(f"[SUCCESS] Models trained and saved in: {cfg.paths.models_dir}")


def cmd_evaluate(args):
    """Executes full benchmark evaluation, generates figures, and validates case studies."""
    runner = SeismicBenchmarkRunner()
    runner.run_benchmark()


def cmd_predict(args):
    """Scores a single earthquake event across all three models."""
    cfg = load_config()
    service = TsunamiInferenceService(cfg)
    pred = service.predict_event(
        magnitude=args.mag,
        depth=args.depth,
        latitude=args.lat,
        longitude=args.lon
    )
    print("\n" + "=" * 60)
    print("  EARTHQUAKE TSUNAMI RISK ASSESSMENT")
    print("=" * 60)
    print(f"  Input: Magnitude={pred.magnitude}, Depth={pred.depth} km, Lat={pred.latitude}, Lon={pred.longitude}")
    print("-" * 60)
    print(f"  Logistic Regression Prob : {pred.lr_probability * 100:.2f}% ({'Tsunami' if pred.lr_prediction == 1 else 'No Tsunami'})")
    print(f"  Support Vector Machine   : {pred.svm_probability * 100:.2f}% ({'Tsunami' if pred.svm_prediction == 1 else 'No Tsunami'})")
    print(f"  Random Forest Prob       : {pred.rf_probability * 100:.2f}% ({'Tsunami' if pred.rf_prediction == 1 else 'No Tsunami'})")
    print("-" * 60)
    print(f"  CONSENSUS ASSESSMENT    : {pred.consensus_risk}")
    print(f"  RISK LEVEL              : {pred.risk_level}")
    print("=" * 60)


def cmd_serve(args):
    """Starts the interactive FastAPI early warning web server."""
    import uvicorn
    from src.api.app import app
    print(f"[START] Launching Tsunami Early Warning Web Dashboard at http://localhost:{args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Driven Classification of Tsunami-Generating Earthquakes (IEEE JSTARS Replication)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # data
    data_parser = subparsers.add_parser("data", help="Harvest USGS data and run preprocessing")
    data_parser.add_argument("--refresh", action="store_true", help="Force refresh data from USGS API")
    data_parser.set_defaults(func=cmd_data)

    # eda
    eda_parser = subparsers.add_parser("eda", help="Generate EDA figures (Fig 1, 2, 3)")
    eda_parser.set_defaults(func=cmd_eda)

    # train
    train_parser = subparsers.add_parser("train", help="Train LR, SVM, and Random Forest models")
    train_parser.add_argument("--no-tune", action="store_true", help="Skip GridSearch and use default RF parameters")
    train_parser.set_defaults(func=cmd_train)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run benchmark evaluation and generate paper figures")
    eval_parser.set_defaults(func=cmd_evaluate)

    # predict
    pred_parser = subparsers.add_parser("predict", help="Predict tsunami risk for a specific earthquake event")
    pred_parser.add_argument("--mag", type=float, required=True, help="Earthquake magnitude (Richter / Mw)")
    pred_parser.add_argument("--depth", type=float, required=True, help="Hypocenter depth in km")
    pred_parser.add_argument("--lat", type=float, required=True, help="Epicenter latitude (-90 to 90)")
    pred_parser.add_argument("--lon", type=float, required=True, help="Epicenter longitude (-180 to 180)")
    pred_parser.set_defaults(func=cmd_predict)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Launch interactive web dashboard")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    serve_parser.set_defaults(func=cmd_serve)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
