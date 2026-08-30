"""Guards docs/index.html's in-browser forward pass.

``js_build_vector`` and ``js_forward`` are line-for-line Python ports of the
JavaScript in ``docs/index.html`` (the submit handler and ``predict()``). They
must agree with ``webapp/pricing.py``; keep the three in sync.
"""

import json
from pathlib import Path

import pytest

from pricing import predict_price

MODEL_JSON = Path(__file__).resolve().parents[1] / "docs" / "model.json"


def load_model():
    if not MODEL_JSON.exists():
        pytest.skip("docs/model.json not generated")
    return json.loads(MODEL_JSON.read_text(encoding="utf-8"))


def js_build_vector(m, feats):
    numeric = m.get("numeric_features") or m.get("features") or []
    medians = m.get("feature_medians", {})
    vec = [float(feats.get(n, medians.get(n, 0.0))) for n in numeric]
    vec += [1.0 if feats.get("Distrito") == c else 0.0 for c in m.get("district_categories", [])]
    return vec


def js_forward(m, values):
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
    n = len(m["numeric_features"]) + len(m["district_categories"])
    assert len(m["scaler"]["mean"]) == n == len(m["scaler"]["scale"])
    assert len(m["layers"]) >= 2
    assert len(m["layers"][0]["W"]) == n
    assert m["layers"][-1]["activation"] == "linear"
    assert m["band"]["low"] < m["band"]["high"]


def test_js_port_matches_numpy_impl():
    m = load_model()
    districts = m.get("district_categories") or [None]
    for district in (districts[0], districts[-1], "Narnia"):
        feats = {**m["feature_medians"], "Distrito": district, "Superficie": 110}
        assert js_forward(m, js_build_vector(m, feats)) == pytest.approx(predict_price(m, feats), rel=1e-6)


def test_bigger_flat_costs_more():
    m = load_model()
    district = (m.get("district_categories") or [None])[0]
    small = {**m["feature_medians"], "Distrito": district, "Superficie": 60}
    big = {**small, "Superficie": 180}
    assert predict_price(m, big) > predict_price(m, small)
