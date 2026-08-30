"""Shared dataset loading for the price-prediction scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Precio"

# ``precio == precio_m2 * superficie`` almost exactly, so Precio_m2 leaks the
# target and is excluded from the honest feature set.
LEAKY_FEATURE = "Precio_m2"
FEATURES = ["Habitaciones", "Aseos", "Superficie", "Parking", "Colegios"]


class DatasetError(Exception):
    """Raised when a dataset CSV is missing required columns."""


def read_csv(csv_route: str) -> pd.DataFrame:
    """Read a CSV whether it was saved as UTF-8 (new) or latin-1 (legacy)."""
    try:
        return pd.read_csv(csv_route, header=0, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_route, header=0, encoding="latin1")


def load_xy(csv_route: str, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for ``features`` and :data:`TARGET`.

    Missing feature values are filled with the column median (not a magic ``1``).
    Raises :class:`DatasetError` if a required column is absent.
    """
    raw = read_csv(csv_route)
    missing = [c for c in [*features, TARGET] if c not in raw.columns]
    if missing:
        raise DatasetError(f"Dataset {csv_route} is missing columns: {missing}")
    cols = raw[[*features, TARGET]].apply(pd.to_numeric, errors="coerce")
    cols = cols.fillna(cols.median(numeric_only=True))
    X = cols[features].astype("float32").to_numpy()
    y = cols[TARGET].astype("float32").to_numpy()
    return X, y


def feature_medians(csv_route: str, features: list[str]) -> dict[str, float]:
    """Column-wise medians of ``features`` (used as defaults at inference time)."""
    raw = read_csv(csv_route)
    cols = raw[features].apply(pd.to_numeric, errors="coerce")
    return {name: float(cols[name].median()) for name in features}
