"""Unit and API tests for Z-score and robust MAD anomaly detection."""

from fastapi.testclient import TestClient

from anomaly.api import app
from anomaly.detector import AnomalyDetector


def test_spike_detection() -> None:
    d = AnomalyDetector()
    res = d.detect_spikes([10.0, 10.2, 10.1, 9.9, 100.5, 10.0], z_threshold=2.0)
    assert len(res["anomalies"]) == 1
    assert res["anomalies"][0]["value"] == 100.5
    assert res["anomalies"][0]["direction"] == "spike"


def test_constant_series_has_zero_false_positives() -> None:
    d = AnomalyDetector()
    res = d.detect_spikes([42.0, 42.0, 42.0, 42.0, 42.0], z_threshold=2.5)
    assert res["std"] == 0.0
    assert res["anomalies"] == []


def test_mad_avoids_extreme_outlier_masking() -> None:
    # A massive spike (10000) inflates std so standard Z-score misses the secondary spike (85)
    series = [10.0, 10.1, 9.9, 10.2, 10.0, 9.8, 85.0, 10000.0]
    d = AnomalyDetector()
    mad_res = d.detect_spikes(series, z_threshold=3.5, method="mad")
    found_values = [item["value"] for item in mad_res["anomalies"]]
    assert 85.0 in found_values
    assert 10000.0 in found_values


def test_api_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "healthy"}

    resp = client.post(
        "/api/v1/detect",
        json={"values": [5.0, 5.1, 4.9, 5.0, 1.0], "threshold": 1.8, "method": "mad"},
    )
    assert resp.status_code == 200
    assert resp.json()["anomalies"][0]["direction"] == "drop"
