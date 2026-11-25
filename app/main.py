import json
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict


# --------------------------------------------------
# 1. Load champion model
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 2. Configure structured JSON logging
# --------------------------------------------------

class JSONFormatter(logging.Formatter):

    def format(self, record):

        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "event": record.getMessage(),
            "prediction": getattr(record, "prediction", None),
            "latency_ms": getattr(record, "latency_ms", None)
        })


logger = logging.getLogger("mp233_api")
logger.setLevel(logging.INFO)
logger.propagate = False

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())

logger.handlers.clear()
logger.addHandler(handler)


# --------------------------------------------------
# 3. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="MP233 Used Two-Wheeler Price Prediction",
    description="G164 - MLOps Mini Project",
    version="1.0"
)


# --------------------------------------------------
# 4. Request schema and client validation rules
# --------------------------------------------------

class PredictionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    age_years: float = Field(..., le=30)
    km_driven: int
    engine_cc: int = Field(..., ge=50)
    owners: int
    service_records: int
    accidents: int
    brand_tier: Literal["budget", "mid", "premium"]
    city_zone: Literal["North", "South", "Central"]


# --------------------------------------------------
# 5. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {"status": "ok"}


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.perf_counter()

    input_data = pd.DataFrame([
        request.model_dump()
    ])

    prediction = float(model.predict(input_data)[0])

    latency_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    logger.info(
        "prediction",
        extra={
            "prediction": prediction,
            "latency_ms": latency_ms
        }
    )

    return {
        "prediction": prediction
    }
