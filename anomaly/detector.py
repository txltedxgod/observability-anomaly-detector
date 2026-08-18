import numpy as np
from typing import List, Dict, Any

class AnomalyDetector:
    def detect_spikes(self, values: List[float], z_threshold: float = 2.5) -> Dict[str, Any]:
        if len(values) < 3:
            return {"anomalies": []}
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr) or 1e-6
        z_scores = (arr - mean) / std

        anomalies = []
        for idx, z in enumerate(z_scores):
            if abs(z) >= z_threshold:
                anomalies.append({"index": idx, "value": values[idx], "z_score": round(float(z), 2)})

        return {
            "metric_count": len(values),
            "mean": round(float(mean), 2),
            "std": round(float(std), 2),
            "anomalies": anomalies
        }
