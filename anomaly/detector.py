"""Statistical anomaly detector supporting Z-score and robust MAD algorithms."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np


class AnomalyDetector:
    """Detect metric spikes/drops via standard Z-score or Modified Z-score (MAD)."""

    def detect_spikes(
        self,
        values: list[float],
        z_threshold: float = 2.5,
        method: Literal["zscore", "mad"] = "zscore",
    ) -> dict[str, Any]:
        """Identify anomalous points in a numeric time series."""
        if z_threshold <= 0:
            raise ValueError("z_threshold must be strictly positive.")
        if any(not math.isfinite(v) for v in values):
            raise ValueError("values must contain only finite numbers.")

        if len(values) < 3:
            return {
                "metric_count": len(values),
                "method": method,
                "mean": round(float(np.mean(values)), 4) if values else 0.0,
                "std": 0.0,
                "anomalies": [],
            }

        arr = np.asarray(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))

        if method == "mad":
            median = float(np.median(arr))
            abs_dev = np.abs(arr - median)
            mad = float(np.median(abs_dev))
            scale = 1.4826 * mad
            scores = (arr - median) / scale if scale > 1e-9 else np.zeros_like(arr)
        else:
            scores = (arr - mean) / std if std > 1e-9 else np.zeros_like(arr)

        anomalies: list[dict[str, Any]] = []
        for idx, score in enumerate(scores):
            if abs(float(score)) >= z_threshold:
                anomalies.append(
                    {
                        "index": idx,
                        "value": values[idx],
                        "z_score": round(float(score), 4),
                        "direction": "spike" if score > 0 else "drop",
                    }
                )

        return {
            "metric_count": len(values),
            "method": method,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "anomalies": anomalies,
        }
