"""The agent's tools - pure functions, no Streamlit or Gemini imports.

Both the Streamlit app and the manual-loop CLI (`agent_manual.py`) register
these with the model; `tests/test_tools.py` exercises them directly.
"""

from __future__ import annotations

from pricing import (
    district_schools,
    estimate_by_district,
    load_district_schools,
    load_districts,
    load_model,
    predict_price,
)

_MODEL = load_model()
_DISTRICTS = load_districts()
_SCHOOLS = load_district_schools()

VALID_DISTRICTS = sorted(_DISTRICTS)


def estimate_price(district: str, rooms: int, bathrooms: int, area_m2: float, parking: int) -> dict:
    """Estimate the sale price of one flat in Madrid.

    Args:
        district: Madrid district (e.g. Salamanca, Chamberí, Carabanchel).
        rooms: number of rooms.
        bathrooms: number of bathrooms.
        area_m2: floor area in square metres.
        parking: 1 if it has a parking space, 0 otherwise.
    """
    by_district = estimate_by_district(_DISTRICTS, district, area_m2, parking)
    schools = district_schools(_SCHOOLS, district)
    nn_price = predict_price(
        _MODEL,
        {
            "Distrito": by_district["distrito"],
            "Habitaciones": rooms,
            "Aseos": bathrooms,
            "Superficie": area_m2,
            "Parking": parking,
            "Colegios": schools,
        },
    )
    return {
        "price_eur": by_district["price_eur"],
        "method": f"{by_district['distrito']} at {by_district['eur_m2']:,} €/m² (~2024)",
        "schools_by_district": schools,
        "reference_neural_network_2020_eur": round(nn_price),
    }


def compare_districts(area_m2: float, parking: int = 0) -> dict:
    """Price the same flat in every Madrid district, cheapest first.

    Args:
        area_m2: floor area in square metres.
        parking: 1 if it has a parking space, 0 otherwise.
    """
    rows = [
        {
            "district": name,
            "price_eur": estimate_by_district(_DISTRICTS, name, area_m2, parking)["price_eur"],
            "eur_m2": round(_DISTRICTS[name]),
        }
        for name in _DISTRICTS
    ]
    rows.sort(key=lambda r: r["price_eur"])
    return {"area_m2": area_m2, "parking": bool(parking), "by_district": rows}


TOOLS = [estimate_price, compare_districts]
