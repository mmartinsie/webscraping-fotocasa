from bs4 import BeautifulSoup

from webscraping.home import Home
from webscraping.listing import _parse_characteristics, _parse_header_features, _parse_price

PRICE_HTML = '<span class="re-DetailHeader-price">234.000 &euro;</span>'

HEADER_HTML = """
<ul>
  <li class="re-DetailHeader-featuresItem"><span>x</span><span>3 hab.</span><span>i</span></li>
  <li class="re-DetailHeader-featuresItem"><span>x</span><span>2 ba&ntilde;os</span><span>i</span></li>
  <li class="re-DetailHeader-featuresItem"><span>x</span><span>90 m&sup2;</span><span>i</span></li>
  <li class="re-DetailHeader-featuresItem"><span>x</span><span>4 Planta</span><span>i</span></li>
</ul>
"""

FEATURES_HTML = """
<div>
  <div class="re-DetailFeaturesList-feature">
    <p class="re-DetailFeaturesList-featureLabel">Tipo de inmueble</p>
    <p class="re-DetailFeaturesList-featureValue">&Aacute;tico</p>
  </div>
  <div class="re-DetailFeaturesList-feature">
    <p class="re-DetailFeaturesList-featureLabel">Parking</p>
    <p class="re-DetailFeaturesList-featureValue">S&iacute;</p>
  </div>
</div>
"""


def test_parse_price():
    soup = BeautifulSoup(PRICE_HTML, "html.parser")
    assert _parse_price(soup) == 234000


def test_parse_price_missing_returns_none():
    assert _parse_price(BeautifulSoup("<div/>", "html.parser")) is None


def test_parse_header_features():
    home = Home(url="u", district="d")
    _parse_header_features(BeautifulSoup(HEADER_HTML, "html.parser"), home)
    assert (home.rooms, home.baths, home.size, home.floor) == (3, 2, 90, 4)


def test_parse_characteristics():
    home = Home(url="u", district="d")
    _parse_characteristics(BeautifulSoup(FEATURES_HTML, "html.parser"), home)
    assert home.property_type == "Ático"
    assert home.parking is True


def test_parse_characteristics_no_parking():
    html = FEATURES_HTML.replace("Parking", "Orientación")
    home = Home(url="u", district="d")
    _parse_characteristics(BeautifulSoup(html, "html.parser"), home)
    assert home.parking is False
