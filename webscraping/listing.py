"""Parse a single Fotocasa listing (detail page) into a :class:`Home`.

The CSS classes below (``re-DetailHeader-*``, ``re-DetailFeaturesList-*``) reflect
Fotocasa's markup at the time the thesis was written and are likely to have
changed since; update them here if the parser stops finding data.
"""

from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

from home import Home

logger = logging.getLogger(__name__)

PRICE_CLASS = "re-DetailHeader-price"
HEADER_FEATURE_CLASS = "re-DetailHeader-featuresItem"
FEATURE_CLASS = "re-DetailFeaturesList-feature"
FEATURE_LABEL_CLASS = "re-DetailFeaturesList-featureLabel"
FEATURE_VALUE_CLASS = "re-DetailFeaturesList-featureValue"

REQUEST_TIMEOUT = 30


def _digits(text: str) -> int | None:
    """Return the integer formed by the digits in ``text`` (``None`` if none)."""
    digits = re.sub(r"\D", "", text or "")
    return int(digits) if digits else None


def _parse_price(soup: BeautifulSoup) -> int | None:
    node = soup.find("span", class_=PRICE_CLASS)
    return _digits(node.get_text()) if node else None


def _parse_header_features(soup: BeautifulSoup, home: Home) -> None:
    """Fill ``rooms``/``baths``/``size``/``floor`` from the header feature list."""
    for item in soup.find_all("li", class_=HEADER_FEATURE_CLASS):
        spans = item.find_all("span")
        if len(spans) < 2:
            continue
        parts = spans[-2].get_text().split()
        if len(parts) < 2:
            continue
        number, unit = parts[0], parts[1]
        if unit in ("hab.", "habs."):
            home.rooms = _digits(number)
        elif unit in ("baño", "baños"):
            home.baths = _digits(number)
        elif unit.startswith("m"):
            home.size = _digits(number)
        elif unit == "Planta":
            home.floor = _digits(number)


def _parse_characteristics(soup: BeautifulSoup, home: Home) -> None:
    """Fill ``property_type``/``parking`` from the detail feature list."""
    for feature in soup.find_all("div", class_=FEATURE_CLASS):
        label_node = feature.find("p", class_=FEATURE_LABEL_CLASS)
        value_node = feature.find("p", class_=FEATURE_VALUE_CLASS)
        if not label_node:
            continue
        label = label_node.get_text().strip()
        if label == "Tipo de inmueble" and value_node:
            home.property_type = value_node.get_text().strip()
        elif label == "Parking":
            home.parking = True


def scrape_listing(session: requests.Session, url: str, district: str) -> Home | None:
    """Download ``url`` and return the parsed :class:`Home`, or ``None`` on failure."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not fetch %s: %s", url, exc)
        return None

    soup = BeautifulSoup(response.text, "lxml")
    home = Home(url=url, district=district)
    home.price = _parse_price(soup)
    _parse_header_features(soup, home)
    _parse_characteristics(soup, home)

    logger.debug("Parsed %s", home)
    return home
