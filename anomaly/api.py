from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from anomaly.detector import AnomalyDetector

app = FastAPI(title="Observability Anomaly Detector", version="0.1.0")
detector = AnomalyDetector()

class MetricReq(BaseModel):
    values: List[float]
    threshold: float = 2.5

@app.post("/api/v1/detect")
def detect(req: MetricReq):
    return detector.detect_spikes(req.values, req.threshold)
