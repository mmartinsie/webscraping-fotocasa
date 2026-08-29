"""Shared dataset loading for the price-prediction scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Precio"

# ``precio == precio_m2 * superficie`` almost exactly, so Precio_m2 leaks the
# target and is excluded from the honest feature set.
LEAKY_FEATURE = "Precio_m2"
FEATURES = ["Habitaciones", "Aseos", "Superficie", "Parking", "Colegios"]


def load_xy(csv_route: str, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for ``features`` and :data:`TARGET`, NaNs filled with 1."""
    raw = pd.read_csv(csv_route, header=0, encoding="latin1")
    data = raw.fillna(value=1)
    missing = [c for c in [*features, TARGET] if c not in data.columns]
    if missing:
        raise SystemExit(f"Dataset {csv_route} is missing columns: {missing}")
    X = data[features].astype("float32").to_numpy()
    y = data[TARGET].astype("float32").to_numpy()
    return X, y
