"""
FastAPI Backend Application for Tsunami Early Warning System
Exposes REST endpoints for live model scoring, metrics, historical case studies, and web frontend.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import load_config, get_project_root
from src.inference.predict import TsunamiInferenceService, TsunamiRiskPrediction


class EarthquakeEventRequest(BaseModel):
    """Input seismic parameters for real-time tsunami risk scoring."""
    magnitude: float = Field(..., ge=1.0, le=10.0, description="Magnitude on Richter / Mw scale", example=8.1)
    depth: float = Field(..., ge=0.0, le=800.0, description="Hypocenter depth in kilometers", example=18.0)
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Epicenter latitude in degrees", example=-15.489)
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Epicenter longitude in degrees", example=-172.095)


class HistoricalEvent(BaseModel):
    name: str
    date: str
    magnitude: float
    depth: float
    latitude: float
    longitude: float
    actual_tsunami: int
    description: str


# Initialize FastAPI app
app = FastAPI(
    title="AI-Driven Tsunami Early Warning System",
    description="Early detection and binary classification of tsunami-generating earthquakes using LR, SVM, and Random Forest.",
    version="1.0.0",
)

# Enable CORS for local web interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root directory & paths
root_dir = get_project_root()
frontend_dir = root_dir / "frontend"
reports_dir = root_dir / "reports"
figures_dir = reports_dir / "figures"

# Initialize global inference service
inference_service: Optional[TsunamiInferenceService] = None


@app.on_event("startup")
def startup_event():
    """Initializes and pre-warms ML models upon server startup."""
    global inference_service
    cfg = load_config()
    inference_service = TsunamiInferenceService(cfg)
    print("[API] Tsunami Early Warning Inference Engine initialized successfully.")


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns system status and loaded models."""
    return {
        "status": "operational",
        "system": "AI-Driven Tsunami Early Warning System",
        "models": ["Logistic Regression", "Support Vector Machine", "Random Forest"],
        "dataset_scope": "USGS 2015-2025 Global Earthquakes (M >= 6.0)",
    }


@app.post("/api/predict")
def predict_tsunami_risk(event: EarthquakeEventRequest) -> Dict[str, Any]:
    """
    Computes real-time tsunami probability and classification across all 3 models.
    """
    global inference_service
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready.")

    prediction: TsunamiRiskPrediction = inference_service.predict_event(
        magnitude=event.magnitude,
        depth=event.depth,
        latitude=event.latitude,
        longitude=event.longitude,
    )

    return {
        "magnitude": prediction.magnitude,
        "depth": prediction.depth,
        "latitude": prediction.latitude,
        "longitude": prediction.longitude,
        "models": {
            "logistic_regression": {
                "name": "Logistic Regression (Linear Baseline)",
                "probability": prediction.lr_probability,
                "prediction": prediction.lr_prediction,
                "prediction_label": "Tsunami" if prediction.lr_prediction == 1 else "No Tsunami",
            },
            "support_vector_machine": {
                "name": "Support Vector Machine (RBF Kernel)",
                "probability": prediction.svm_probability,
                "prediction": prediction.svm_prediction,
                "prediction_label": "Tsunami" if prediction.svm_prediction == 1 else "No Tsunami",
            },
            "random_forest": {
                "name": "Random Forest (Ensemble Optimizer)",
                "probability": prediction.rf_probability,
                "prediction": prediction.rf_prediction,
                "prediction_label": "Tsunami" if prediction.rf_prediction == 1 else "No Tsunami",
            },
        },
        "consensus_risk": prediction.consensus_risk,
        "risk_level": prediction.risk_level,
    }


@app.get("/api/case-studies", response_model=List[HistoricalEvent])
def get_case_studies() -> List[HistoricalEvent]:
    """Returns curated historical earthquake events for instant demonstration."""
    return [
        HistoricalEvent(
            name="2009 Samoa Earthquake (Paper Benchmark)",
            date="2009-09-29",
            magnitude=8.1,
            depth=18.0,
            latitude=-15.489,
            longitude=-172.095,
            actual_tsunami=1,
            description="Major submarine thrust fault event near the Tonga Trench triggering devastating Pacific tsunami waves.",
        ),
        HistoricalEvent(
            name="2011 Tohoku Great East Japan Earthquake",
            date="2011-03-11",
            magnitude=9.1,
            depth=29.0,
            latitude=38.297,
            longitude=142.372,
            actual_tsunami=1,
            description="Megathrust earthquake in the Japan Trench causing massive ocean floor vertical displacement.",
        ),
        HistoricalEvent(
            name="2004 Indian Ocean Earthquake (Sumatra-Andaman)",
            date="2004-12-26",
            magnitude=9.1,
            depth=30.0,
            latitude=3.295,
            longitude=95.982,
            actual_tsunami=1,
            description="Subduction megathrust event in the Sunda Trench generating a catastrophic basin-wide Indian Ocean tsunami.",
        ),
        HistoricalEvent(
            name="2023 Turkey-Syria Inland Earthquake",
            date="2023-02-06",
            magnitude=7.8,
            depth=10.0,
            latitude=37.166,
            longitude=37.032,
            actual_tsunami=0,
            description="Inland strike-slip continental fault event with no submarine ocean water column displacement.",
        ),
        HistoricalEvent(
            name="2024 Noto Peninsula Earthquake (Japan)",
            date="2024-01-01",
            magnitude=7.5,
            depth=10.0,
            latitude=37.500,
            longitude=137.240,
            actual_tsunami=1,
            description="Shallow crustal coastal reverse fault earthquake that generated localized tsunami waves in the Sea of Japan.",
        ),
    ]


@app.get("/api/metrics")
def get_metrics_summary() -> Dict[str, Any]:
    """Returns quantitative benchmarking metrics."""
    metrics_file = reports_dir / "metrics_summary.json"
    if metrics_file.exists():
        import json
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Run 'python main.py evaluate' to generate metrics summary."}


# Mount static assets if directories exist
if figures_dir.exists():
    app.mount("/reports/figures", StaticFiles(directory=str(figures_dir)), name="figures")

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend_static")

    @app.get("/")
    def serve_frontend_index():
        return FileResponse(frontend_dir / "index.html")
