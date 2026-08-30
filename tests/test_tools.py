import inspect

import pytest

from tools import TOOLS, VALID_DISTRICTS, compare_districts, estimate_price


def test_tools_registered():
    assert {fn.__name__ for fn in TOOLS} == {"estimate_price", "compare_districts"}


def test_every_tool_param_is_documented():
    # A tool the model can't read the docstring of is a tool it will misuse.
    for fn in TOOLS:
        doc = inspect.getdoc(fn) or ""
        for name in inspect.signature(fn).parameters:
            assert name in doc, f"{fn.__name__}: '{name}' missing from docstring"


def test_estimate_price_shape():
    out = estimate_price("Salamanca", 3, 2, 90, 0)
    assert {"district", "price_eur", "method", "schools_by_district"} <= set(out)
    assert "error" not in out
    assert out["district"] == "Salamanca"
    assert out["price_eur"] > 0
    assert out["schools_by_district"] == 3  # from data/colegios_distrito.csv


def test_estimate_price_rejects_bad_inputs():
    assert "error" in estimate_price("Salamanca", 3, 2, 0, 0)  # zero area
    assert "error" in estimate_price("Salamanca", 3, 2, -50, 0)  # negative area
    assert "error" in estimate_price("Salamanca", 0, 2, 90, 0)  # no rooms
    unknown = estimate_price("Gotham", 3, 2, 90, 0)
    assert "error" in unknown and "Salamanca" in unknown["valid_districts"]


def test_estimate_price_accepts_fuzzy_district():
    assert estimate_price("chamberi", 3, 2, 90, 0)["district"] == "Chamberí"
    assert estimate_price("puente vallecas", 3, 2, 90, 0)["district"] == "Puente de Vallecas"


def test_estimate_price_scales_with_area_and_parking():
    base = estimate_price("Retiro", 3, 2, 80, 0)["price_eur"]
    assert estimate_price("Retiro", 3, 2, 160, 0)["price_eur"] == pytest.approx(2 * base, rel=1e-6)
    assert estimate_price("Retiro", 3, 2, 80, 1)["price_eur"] > base


def test_compare_districts_is_sorted_and_complete():
    out = compare_districts(90, 0)
    rows = out["by_district"]
    assert [r["district"] for r in rows] and len(rows) == len(VALID_DISTRICTS)
    assert [r["price_eur"] for r in rows] == sorted(r["price_eur"] for r in rows)
    assert rows[0]["district"] == "Villaverde"  # cheapest €/m²
    assert rows[-1]["district"] == "Salamanca"  # most expensive


def test_compare_districts_rejects_bad_area():
    assert "error" in compare_districts(0)
    assert "error" in compare_districts(-10)
