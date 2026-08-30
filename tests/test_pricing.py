import json
from pathlib import Path

import pytest

from pricing import (
    district_eur_m2,
    district_schools,
    estimate_by_district,
    load_district_schools,
    load_districts,
    load_model,
    match_district,
    predict_price,
)

MODEL_JSON = Path(__file__).resolve().parents[1] / "docs" / "model.json"


@pytest.fixture
def model():
    if not MODEL_JSON.exists():
        pytest.skip("docs/model.json not generated")
    return load_model(MODEL_JSON)


def test_load_model_shape(model):
    n = len(model["numeric_features"]) + len(model["district_categories"])
    assert n > 0
    assert len(model["scaler"]["mean"]) == n


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


# --- district €/m² estimate ------------------------------------------------- #


@pytest.fixture
def districts():
    return load_districts()


def test_districts_table_has_all_madrid_districts(districts):
    assert len(districts) == 21
    assert "Salamanca" in districts and "Villaverde" in districts


def test_district_lookup_is_accent_and_case_insensitive(districts):
    assert district_eur_m2(districts, "chamberi")[1] == "Chamberí"
    assert district_eur_m2(districts, "SALAMANCA")[1] == "Salamanca"


def test_match_district_tokens_and_rejects_ambiguous(districts):
    assert match_district(districts, "puente vallecas") == "Puente de Vallecas"
    assert match_district(districts, "ciudad-lineal".replace("-", " ")) == "Ciudad Lineal"
    assert match_district(districts, "vallecas") is None  # ambiguous, one word
    assert match_district(districts, "Gotham") is None
    assert match_district(districts, None) is None


def test_unknown_district_falls_back_to_average(districts):
    value, name = district_eur_m2(districts, "Narnia")
    assert name == "media de Madrid"
    assert value == pytest.approx(sum(districts.values()) / len(districts))


def test_estimate_scales_with_size_and_parking(districts):
    base = estimate_by_district(districts, "Salamanca", 90, parking=0)["price_eur"]
    assert estimate_by_district(districts, "Salamanca", 180, parking=0)["price_eur"] == pytest.approx(
        2 * base, rel=1e-6
    )
    assert estimate_by_district(districts, "Salamanca", 90, parking=1)["price_eur"] > base


def test_expensive_district_costs_more(districts):
    salamanca = estimate_by_district(districts, "Salamanca", 90)["price_eur"]
    villaverde = estimate_by_district(districts, "Villaverde", 90)["price_eur"]
    assert salamanca > 2 * villaverde


# --- schools automated from district -------------------------------------- #


def test_school_table_matches_district_table():
    assert set(load_district_schools()) == set(load_districts())


def test_district_schools_lookup():
    schools = load_district_schools()
    assert district_schools(schools, "Salamanca") == 3
    assert district_schools(schools, "puente de vallecas") == 25


def test_unknown_district_schools_falls_back_to_average():
    schools = load_district_schools()
    expected = round(sum(schools.values()) / len(schools))
    assert district_schools(schools, None) == expected
    assert district_schools(schools, "Narnia") == expected
