"""Stage 3: FastAPI service for ATM cash demand prediction.

Endpoints:
  GET  /health   - liveness / readiness check
  POST /predict  - predict ATM cash withdrawal demand
"""

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields attached by the caller
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info", "exc_text",
                "stack_info", "lineno", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "message", "taskName",
            }:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = _build_logger("mp293.api")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

_model = None  # loaded lazily on first request


def get_model():
    """Return the cached champion model, loading it on first call."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            log.error(
                "Model artifact not found",
                extra={"model_path": str(MODEL_PATH)},
            )
            raise RuntimeError(
                f"Model artifact not found at {MODEL_PATH}. "
                "Export the champion model from MLflow first."
            )
        try:
            _model = joblib.load(MODEL_PATH)
            log.info(
                "Model loaded successfully",
                extra={"model_path": str(MODEL_PATH)},
            )
        except Exception as exc:
            log.error(
                "Failed to load model",
                extra={"model_path": str(MODEL_PATH), "error": str(exc)},
            )
            raise RuntimeError(f"Cannot load model: {exc}") from exc
    return _model


# ---------------------------------------------------------------------------
# Preprocessing helpers (mirrors training pipeline)
# ---------------------------------------------------------------------------
# Expected one-hot columns produced by training preprocessor
# (categories sorted alphabetically as in fit_preprocessor)
ATM_TYPE_CATEGORIES = ["bank_branch", "mall", "standalone"]
AREA_CATEGORIES = ["highway", "market", "residential"]

NUMERIC_FEATURES = [
    "nearby_shops",
    "days_to_salary",
    "footfall_index",
    "distance_to_market_km",
    "avg_ticket_rs",
    "other_atms_1km",
]

# Final feature columns in the exact order the model was trained on
FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + [f"atm_type_{c}" for c in ATM_TYPE_CATEGORIES]
    + [f"area_{c}" for c in AREA_CATEGORIES]
)

# Fallback medians if preprocessor.json is unavailable
_FALLBACK_MEDIANS: dict[str, float] = {
    "nearby_shops": 15.0,
    "days_to_salary": 10.0,
    "footfall_index": 55.0,
    "distance_to_market_km": 3.0,
    "avg_ticket_rs": 2000.0,
    "other_atms_1km": 3.0,
}

_preprocessor: dict | None = None


def _load_preprocessor() -> dict:
    global _preprocessor
    if _preprocessor is None:
        pp_path = PROJECT_ROOT / "data" / "processed" / "preprocessor.json"
        if pp_path.exists():
            with open(pp_path, "r", encoding="utf-8") as f:
                _preprocessor = json.load(f)
            log.info("Preprocessor loaded", extra={"path": str(pp_path)})
        else:
            log.warning(
                "preprocessor.json not found, using fallback medians",
                extra={"path": str(pp_path)},
            )
            _preprocessor = {
                "numeric_medians": _FALLBACK_MEDIANS,
                "categorical_categories": {
                    "atm_type": ATM_TYPE_CATEGORIES,
                    "area": AREA_CATEGORIES,
                },
            }
    return _preprocessor


def _build_feature_vector(inp: "PredictRequest") -> pd.DataFrame:
    """Transform the 8 raw API fields into the model's feature vector."""
    pp = _load_preprocessor()
    medians: dict = pp["numeric_medians"]

    # Normalise area to lowercase (matches training)
    area_norm = inp.area.strip().lower()

    # Build a single-row DataFrame matching training feature order
    row: dict[str, float] = {}

    for feat in NUMERIC_FEATURES:
        val = getattr(inp, feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            row[feat] = medians.get(feat, 0.0)
        else:
            row[feat] = float(val)

    # One-hot encode atm_type
    for cat in ATM_TYPE_CATEGORIES:
        row[f"atm_type_{cat}"] = 1 if inp.atm_type == cat else 0

    # One-hot encode area
    for cat in AREA_CATEGORIES:
        row[f"area_{cat}"] = 1 if area_norm == cat else 0

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

VALID_ATM_TYPES = Literal["bank_branch", "mall", "standalone"]
VALID_AREAS = Literal["Highway", "Market", "Residential"]


class PredictRequest(BaseModel):
    nearby_shops: float = Field(..., description="Number of nearby shops")
    days_to_salary: float = Field(..., description="Days until next salary day")
    footfall_index: float = Field(..., description="Daily footfall index")
    distance_to_market_km: float = Field(
        ..., ge=0, description="Distance to nearest market (km)"
    )
    avg_ticket_rs: float = Field(
        ..., ge=250, description="Average ticket size in Rs (minimum 250)"
    )
    other_atms_1km: float = Field(
        ..., ge=0, le=30, description="Competing ATMs within 1 km (max 30)"
    )
    atm_type: VALID_ATM_TYPES = Field(
        ..., description="ATM type: bank_branch | mall | standalone"
    )
    area: VALID_AREAS = Field(
        ..., description="Area type: Highway | Market | Residential"
    )

    model_config = {"extra": "forbid"}


class PredictResponse(BaseModel):
    predicted_withdrawal_thousand_rs: float
    model: str = "atm_cash_demand_model@champion"
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ATM Cash Demand Prediction API",
    description="Stage 3 — G232 / MP293. Predicts ATM cash withdrawal demand.",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    log.warning(
        "Request validation failed",
        extra={
            "endpoint": str(request.url.path),
            "errors": exc.errors(),
        },
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness and readiness probe."""
    model_loaded = MODEL_PATH.exists()
    log.info("Health check", extra={"endpoint": "/health", "model_file_exists": model_loaded})
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        model_path=str(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest):
    """Predict ATM cash withdrawal demand from 8 input features."""
    log.info(
        "Prediction request received",
        extra={
            "endpoint": "/predict",
            "atm_type": request.atm_type,
            "area": request.area,
        },
    )

    try:
        model = get_model()
    except RuntimeError as exc:
        log.error(
            "Model unavailable",
            extra={"endpoint": "/predict", "error": str(exc)},
        )
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        X = _build_feature_vector(request)
        prediction = float(model.predict(X)[0])
    except Exception as exc:
        log.error(
            "Inference error",
            extra={
                "endpoint": "/predict",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )

    log.info(
        "Prediction successful",
        extra={
            "endpoint": "/predict",
            "predicted_withdrawal_thousand_rs": prediction,
            "status": "success",
        },
    )

    return PredictResponse(predicted_withdrawal_thousand_rs=prediction)
