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
    assert set(out) == {
        "price_eur",
        "method",
        "schools_by_district",
        "reference_neural_network_2020_eur",
    }
    assert out["price_eur"] > 0
    assert out["schools_by_district"] == 3  # from data/colegios_distrito.csv


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
