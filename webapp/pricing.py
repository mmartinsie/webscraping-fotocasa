"""Price estimation for the demos.

Two estimates:

- ``estimate_by_district`` - ``€/m² of the district × m²`` using the ~2024-2025
  reference table in ``data/precio_m2_distrito.csv``. This carries the location
  and current-price-level signal.
- ``predict_price`` - the thesis neural network from ``docs/model.json``, run in
  NumPy (no TensorFlow, fast cold start). Trained on ~2020 data with the district
  dropped; kept as a secondary reference number.

``Colegios`` (nearby schools) is a district-level attribute in the thesis data,
so it is looked up from ``data/colegios_distrito.csv`` rather than asked for.

The NumPy forward pass matches ``keras_neural_network/predict.py`` and the
JavaScript in ``docs/index.html``.
"""

from __future__ import annotations

import csv
import json
import unicodedata
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_JSON = _ROOT / "docs" / "model.json"
DEFAULT_DISTRICT_CSV = _ROOT / "data" / "precio_m2_distrito.csv"
DEFAULT_SCHOOLS_CSV = _ROOT / "data" / "colegios_distrito.csv"

PARKING_PREMIUM = 1.06  # a parking space adds roughly 6% to the price


# --------------------------------------------------------------------------- #
# Per-district reference tables
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text.strip().lower()


def _load_table(path: str | Path, value_col: str) -> dict[str, float]:
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["Distrito"]: float(row[value_col]) for row in csv.DictReader(handle)}


def match_district(table: dict[str, float], distrito: str | None) -> str | None:
    """Canonical district name for a fuzzy input, or ``None`` if unrecognised.

    Exact match after accent/case folding, or (for multi-word inputs) all of the
    input's words being a subset of the district's words - so "puente vallecas"
    resolves but a bare "vallecas" (ambiguous) does not.
    """
    if not distrito:
        return None
    target = _norm(distrito)
    tokens = set(target.split())
    for name in table:
        words = set(_norm(name).split())
        if _norm(name) == target or (len(tokens) >= 2 and tokens <= words):
            return name
    return None


def _lookup(table: dict[str, float], distrito: str | None) -> tuple[float, str]:
    """``(value, name)`` for ``distrito``; the table average if unrecognised."""
    name = match_district(table, distrito)
    if name is not None:
        return table[name], name
    return sum(table.values()) / len(table), "media de Madrid"


def load_districts(path: str | Path = DEFAULT_DISTRICT_CSV) -> dict[str, float]:
    """Return ``{district name: €/m²}``."""
    return _load_table(path, "EurM2")


def load_district_schools(path: str | Path = DEFAULT_SCHOOLS_CSV) -> dict[str, float]:
    """Return ``{district name: nearby schools}`` (from the thesis dataset)."""
    return _load_table(path, "Colegios")


def district_eur_m2(districts: dict[str, float], distrito: str | None) -> tuple[float, str]:
    return _lookup(districts, distrito)


def district_schools(schools: dict[str, float], distrito: str | None) -> float:
    """Nearby-school count for ``distrito`` (average if unknown)."""
    return round(_lookup(schools, distrito)[0])


def estimate_by_district(
    districts: dict[str, float], distrito: str | None, superficie: float, parking: int = 0
) -> dict:
    eur_m2, matched = district_eur_m2(districts, distrito)
    price = eur_m2 * float(superficie) * (PARKING_PREMIUM if parking else 1.0)
    return {"price_eur": round(price), "eur_m2": round(eur_m2), "distrito": matched}


# --------------------------------------------------------------------------- #
# Thesis neural network (docs/model.json)
# --------------------------------------------------------------------------- #
def load_model(path: str | Path = DEFAULT_MODEL_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def feature_vector(model: dict, features: dict) -> list[float]:
    """Model input: numeric features (median-filled) then the district one-hot."""
    names = model.get("numeric_features") or model.get("features", [])
    medians = model.get("feature_medians", {})
    row = [float(features.get(n, medians.get(n, 0.0))) for n in names]
    row += [1.0 if features.get("Distrito") == c else 0.0 for c in model.get("district_categories", [])]
    return row


def predict_price(model: dict, features: dict) -> float:
    """Estimate a flat's price with the thesis network."""
    x = np.asarray(feature_vector(model, features), dtype="float64")
    mean = np.asarray(model["scaler"]["mean"], dtype="float64")
    scale = np.asarray(model["scaler"]["scale"], dtype="float64")
    v = (x - mean) / scale
    for layer in model["layers"]:
        v = v @ np.asarray(layer["W"], dtype="float64") + np.asarray(layer["b"], dtype="float64")
        if layer["activation"] == "relu":
            v = np.maximum(v, 0.0)
    price = float(v[0])
    band = model.get("band") or {}
    if band.get("low") is not None:
        price = max(price, band["low"])
    if band.get("high") is not None:
        price = min(price, band["high"])
    return price
