"""FastAPI service exposing statistical anomaly detection endpoints."""

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from anomaly.detector import AnomalyDetector

app = FastAPI(title="Observability Anomaly Detector", version="1.0.0")
detector = AnomalyDetector()


class MetricReq(BaseModel):
    """Time-series anomaly detection request payload."""

    values: list[float]
    threshold: float = Field(default=2.5, gt=0.0, le=20.0)
    method: Literal["zscore", "mad"] = "zscore"


@app.get("/health", tags=["Monitoring"])
def health() -> dict[str, str]:
    """Liveness check endpoint."""
    return {"status": "healthy"}


@app.post("/api/v1/detect", tags=["Detection"])
def detect(req: MetricReq) -> dict[str, Any]:
    """Detect spikes and drops in telemetry series."""
    try:
        return detector.detect_spikes(req.values, req.threshold, method=req.method)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
