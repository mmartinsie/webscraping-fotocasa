"""Price estimation for the demos.

Two estimates:

- ``estimate_by_district`` - ``€/m² of the district × m²`` using the ~2024-2025
  reference table in ``data/precio_m2_distrito.csv``. This carries the location
  and current-price-level signal.
- ``predict_price`` - the thesis neural network from ``docs/model.json``, run in
  NumPy (no TensorFlow, fast cold start). Trained on ~2020 data with the district
  dropped; kept as a secondary reference number.

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

PARKING_PREMIUM = 1.06  # a parking space adds roughly 6% to the price


# --------------------------------------------------------------------------- #
# District €/m² estimate
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text.strip().lower()


def load_districts(path: str | Path = DEFAULT_DISTRICT_CSV) -> dict[str, float]:
    """Return ``{district name: €/m²}`` from the reference CSV."""
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["Distrito"]: float(row["EurM2"]) for row in csv.DictReader(handle)}


def district_eur_m2(districts: dict[str, float], distrito: str | None) -> tuple[float, str]:
    """Look up €/m² for ``distrito`` (accent/case-insensitive). Falls back to the
    table average. Returns ``(€/m², matched name)``."""
    average = sum(districts.values()) / len(districts)
    if not distrito:
        return average, "media de Madrid"
    target = _norm(distrito)
    for name, value in districts.items():
        if _norm(name) == target or target in _norm(name) or _norm(name) in target:
            return value, name
    return average, "media de Madrid"


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


def predict_price(model: dict, features: dict[str, float]) -> float:
    """Estimate a flat's price. Missing features fall back to the stored median."""
    names = model["features"]
    medians = model.get("feature_medians", {})
    x = np.array([float(features.get(name, medians.get(name, 0.0))) for name in names], dtype="float64")
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
