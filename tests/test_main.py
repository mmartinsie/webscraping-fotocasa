import pytest
from main import district_from_label, load_scraped_urls

from home import Home


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Barrio de Salamanca, Madrid Salamanca", "Salamanca"),
        ("Numancia, Puente de Vallecas", "Puente de Vallecas"),
        ("Pueblo Nuevo, Ciudad Lineal", "Ciudad Lineal"),
        ("Simancas, San Blas", "San Blas"),
        ("", None),
    ],
)
def test_district_from_label(text, expected):
    assert district_from_label(text) == expected


def test_district_from_label_strips_prefix():
    # The <span> text is a prefix that must be dropped before tokenising.
    assert district_from_label("desde 200.000 Chamartin", prefix="desde 200.000 ") == "Chamartin"


def test_load_scraped_urls(tmp_path):
    csv = tmp_path / "out.csv"
    csv.write_text("Precio,URL\n100000,http://a\n200000,http://b\n", encoding="utf-8")
    assert load_scraped_urls(str(csv)) == {"http://a", "http://b"}


def test_load_scraped_urls_missing_file(tmp_path):
    assert load_scraped_urls(str(tmp_path / "nope.csv")) == set()


def test_home_is_empty():
    assert Home(url="u", district="d").is_empty()
    assert not Home(url="u", district="d", price=100000).is_empty()
    assert not Home(url="u", district="d", parking=True).is_empty()
