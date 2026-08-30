"""Data model for a single Fotocasa property listing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Single source of truth for the scraper's CSV: (column, how to read it from a Home).
# The Spanish names match the existing ``buildings_information.csv`` and what the
# model scripts expect (they ``drop(["Tipo", "Distrito"])``).
_CSV_FIELDS: list[tuple[str, Callable[[Home], object]]] = [
    ("Precio", lambda h: h.price),
    ("Distrito", lambda h: h.district),
    ("Tipo", lambda h: h.property_type),
    ("Habitaciones", lambda h: h.rooms),
    ("Aseos", lambda h: h.baths),
    ("Superficie", lambda h: h.size),
    ("Planta", lambda h: h.floor),
    ("Parking", lambda h: 1 if h.parking else None),
    ("URL", lambda h: h.url),
]
CSV_HEADERS = [column for column, _ in _CSV_FIELDS]


@dataclass
class Home:
    """A property scraped from Fotocasa.

    ``url`` and ``district`` are known when the object is created (they come from
    the search results page); every other field is filled in later from the
    detail page and stays ``None``/``False`` when the page does not provide it.
    """

    url: str
    district: str
    price: int | None = None
    property_type: str | None = None
    rooms: int | None = None
    baths: int | None = None
    size: int | None = None
    floor: int | None = None
    parking: bool = False

    def is_empty(self) -> bool:
        """True when the detail page yielded nothing (likely a stale selector)."""
        return (
            self.price is None
            and self.property_type is None
            and self.rooms is None
            and self.baths is None
            and self.size is None
            and self.floor is None
            and not self.parking
        )

    def to_csv_row(self) -> dict[str, object]:
        """Return the listing as a dict keyed by :data:`CSV_HEADERS`."""
        return {column: read(self) for column, read in _CSV_FIELDS}
