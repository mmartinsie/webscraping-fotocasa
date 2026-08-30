"""Guards docs/index.html's in-browser forward pass.

``forward`` below is a line-for-line Python port of the JavaScript in
``docs/index.html``. If the network export format or the JS math changes, keep
the two in sync and this test honest.
"""

import json
from pathlib import Path

import pytest

MODEL_JSON = Path(__file__).resolve().parents[1] / "docs" / "model.json"


def load_model():
    if not MODEL_JSON.exists():
        pytest.skip("docs/model.json not generated")
    return json.loads(MODEL_JSON.read_text(encoding="utf-8"))


def forward(m, values):
    v = [(x - m["scaler"]["mean"][i]) / m["scaler"]["scale"][i] for i, x in enumerate(values)]
    for layer in m["layers"]:
        out = []
        for j in range(len(layer["b"])):
            s = layer["b"][j] + sum(v[i] * layer["W"][i][j] for i in range(len(v)))
            out.append(max(s, 0.0) if layer["activation"] == "relu" else s)
        v = out
    price = v[0]
    band = m.get("band") or {}
    if band.get("low") is not None:
        price = max(price, band["low"])
    if band.get("high") is not None:
        price = min(price, band["high"])
    return price


def test_model_json_structure():
    m = load_model()
    n = len(m["features"])
    assert len(m["scaler"]["mean"]) == n == len(m["scaler"]["scale"])
    assert len(m["layers"]) >= 2
    assert m["layers"][0]["W"] and len(m["layers"][0]["W"]) == n
    assert m["layers"][-1]["activation"] == "linear"
    assert m["band"]["low"] < m["band"]["high"]


def test_forward_pass_within_band_and_monotonic_in_size():
    m = load_model()
    medians = m.get("feature_medians")
    typical = [medians[f] for f in m["features"]] if medians else list(m["scaler"]["mean"])
    price = forward(m, typical)
    assert m["band"]["low"] <= price <= m["band"]["high"]

    size_idx = m["features"].index("Superficie")
    bigger = list(typical)
    bigger[size_idx] = bigger[size_idx] * 2 + 1
    assert forward(m, bigger) > price  # more floor area -> higher price
