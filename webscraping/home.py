"""Data model for a single Fotocasa property listing."""

from __future__ import annotations

from dataclasses import dataclass

# CSV column order used by the scraper output. The names are Spanish on purpose:
# they mirror the field labels shown on fotocasa.es.
CSV_HEADERS = [
    "Precio",
    "Distrito",
    "Tipo de inmueble",
    "Habitaciones",
    "Aseos",
    "Superficie",
    "Planta",
    "Parking",
    "URL",
]


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

    def to_csv_row(self) -> dict[str, object]:
        """Return the listing as a dict keyed by :data:`CSV_HEADERS`."""
        return {
            "Precio": self.price,
            "Distrito": self.district,
            "Tipo de inmueble": self.property_type,
            "Habitaciones": self.rooms,
            "Aseos": self.baths,
            "Superficie": self.size,
            "Planta": self.floor,
            "Parking": 1 if self.parking else None,
            "URL": self.url,
        }
