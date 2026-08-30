"""The agent's tools - pure functions, no Streamlit or Gemini imports.

Both the Streamlit app and the manual-loop CLI (`agent_manual.py`) register
these with the model; `tests/test_tools.py` exercises them directly.

Each tool validates its inputs and returns ``{"error": ...}`` on bad ones, so
the model can recover and re-ask instead of getting a silent wrong answer.
"""

from __future__ import annotations

from pricing import (
    district_schools,
    estimate_by_district,
    load_district_schools,
    load_districts,
    load_model,
    match_district,
    predict_price,
)

_MODEL = load_model()
_DISTRICTS = load_districts()
_SCHOOLS = load_district_schools()

VALID_DISTRICTS = sorted(_DISTRICTS)


def _check_area(area_m2: float) -> dict | None:
    if not (0 < float(area_m2) <= 5000):
        return {"error": "area_m2 must be a positive number of square metres (<= 5000)"}
    return None


def estimate_price(district: str, rooms: int, bathrooms: int, area_m2: float, parking: int) -> dict:
    """Estimate the sale price of one flat in Madrid.

    Args:
        district: Madrid district (e.g. Salamanca, Chamberí, Carabanchel).
        rooms: number of rooms.
        bathrooms: number of bathrooms.
        area_m2: floor area in square metres.
        parking: 1 if it has a parking space, 0 otherwise.
    """
    if err := _check_area(area_m2):
        return err
    canonical = match_district(_DISTRICTS, district)
    if canonical is None:
        return {"error": f"unknown district {district!r}", "valid_districts": VALID_DISTRICTS}
    if int(rooms) < 1 or int(bathrooms) < 1:
        return {"error": "rooms and bathrooms must be at least 1"}
    parking = 1 if parking else 0

    by_district = estimate_by_district(_DISTRICTS, canonical, area_m2, parking)
    schools = district_schools(_SCHOOLS, canonical)
    nn_price = predict_price(
        _MODEL,
        {
            "Distrito": canonical,
            "Habitaciones": rooms,
            "Aseos": bathrooms,
            "Superficie": area_m2,
            "Parking": parking,
            "Colegios": schools,
        },
    )
    return {
        "district": canonical,
        "price_eur": by_district["price_eur"],
        "method": f"{canonical} at {by_district['eur_m2']:,} €/m² (~2024)",
        "schools_by_district": schools,
        "reference_neural_network_2020_eur": round(nn_price),
    }


def compare_districts(area_m2: float, parking: int = 0) -> dict:
    """Price the same flat in every Madrid district, cheapest first.

    Args:
        area_m2: floor area in square metres.
        parking: 1 if it has a parking space, 0 otherwise.
    """
    if err := _check_area(area_m2):
        return err
    parking = 1 if parking else 0
    rows = [
        {
            "district": name,
            "price_eur": estimate_by_district(_DISTRICTS, name, area_m2, parking)["price_eur"],
            "eur_m2": round(_DISTRICTS[name]),
        }
        for name in _DISTRICTS
    ]
    rows.sort(key=lambda r: r["price_eur"])
    return {"area_m2": float(area_m2), "parking": bool(parking), "by_district": rows}


TOOLS = [estimate_price, compare_districts]
