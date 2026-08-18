from anomaly.detector import AnomalyDetector

def test_spike_detection():
    d = AnomalyDetector()
    res = d.detect_spikes([10.0, 10.2, 10.1, 9.9, 100.5, 10.0], z_threshold=2.0)
    assert len(res["anomalies"]) >= 1
    assert res["anomalies"][0]["value"] == 100.5
