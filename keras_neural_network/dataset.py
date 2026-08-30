"""Shared dataset loading for the price-prediction scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Precio"
DISTRICT_COL = "Distrito"

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


def one_hot_district(csv_route: str) -> tuple[np.ndarray, list[str]]:
    """Return ``(matrix, categories)`` — a row-aligned one-hot of ``Distrito``.

    ``categories`` is the sorted district list; the same order must be used to
    build the vector at inference time.
    """
    raw = read_csv(csv_route)
    if DISTRICT_COL not in raw.columns:
        raise DatasetError(f"Dataset {csv_route} is missing column: {DISTRICT_COL}")
    series = raw[DISTRICT_COL].astype("string").fillna("")
    categories = sorted(c for c in series.unique() if c)
    matrix = np.array(
        [[1.0 if value == cat else 0.0 for cat in categories] for value in series],
        dtype="float32",
    )
    return matrix, categories


def feature_medians(csv_route: str, features: list[str]) -> dict[str, float]:
    """Column-wise medians of ``features`` (used as defaults at inference time)."""
    raw = read_csv(csv_route)
    cols = raw[features].apply(pd.to_numeric, errors="coerce")
    return {name: float(cols[name].median()) for name in features}
