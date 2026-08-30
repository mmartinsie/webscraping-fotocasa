"""Load ``docs/model.json`` and run its forward pass in NumPy (no TensorFlow).

Same maths as ``keras_neural_network/predict.py`` and the JavaScript in
``docs/index.html`` - kept dependency-free so the Streamlit app has a fast cold
start.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_JSON = Path(__file__).resolve().parents[1] / "docs" / "model.json"


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
