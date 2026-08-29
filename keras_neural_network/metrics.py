"""Regression scoring shared by ``recommend_price.py`` and ``baseline.py``."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, R2 and MAPE (%) for ``y_pred`` against ``y_true``."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100),
    }


def format_row(name: str, m: dict[str, float], width: int = 32) -> str:
    """One aligned ``name  MAE  RMSE  MAPE  R2`` line."""
    return f"{name:<{width}} {m['mae']:>13,.0f} {m['rmse']:>13,.0f} {m['mape']:>7.1f}% {m['r2']:>8.3f}"
