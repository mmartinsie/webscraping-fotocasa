import json
from pathlib import Path

import pytest

from pricing import load_model, predict_price

MODEL_JSON = Path(__file__).resolve().parents[1] / "docs" / "model.json"


@pytest.fixture
def model():
    if not MODEL_JSON.exists():
        pytest.skip("docs/model.json not generated")
    return load_model(MODEL_JSON)


def test_load_model_shape(model):
    assert model["features"]
    assert len(model["scaler"]["mean"]) == len(model["features"])


def test_predict_within_band(model):
    price = predict_price(model, {})  # all-median flat
    assert model["band"]["low"] <= price <= model["band"]["high"]


def test_missing_features_use_median(model):
    # An empty dict and an explicit median dict give the same price.
    medians = model.get("feature_medians") or {}
    assert predict_price(model, {}) == pytest.approx(predict_price(model, dict(medians)))


def test_bigger_flat_costs_more(model):
    medians = model.get("feature_medians") or {}
    base = predict_price(model, {})
    bigger = dict(medians)
    bigger["Superficie"] = bigger.get("Superficie", 90) * 2 + 20
    assert predict_price(model, bigger) > base


def test_matches_committed_golden(model):
    # Regenerate with: cd keras_neural_network && python export_web.py web_model
    # then update this value if the model legitimately changed.
    golden = json.loads(MODEL_JSON.read_text(encoding="utf-8")).get("_golden_median_price")
    if golden is None:
        pytest.skip("no golden stored in model.json")
    assert predict_price(model, {}) == pytest.approx(golden, abs=1.0)
