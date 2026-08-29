import pytest
from main import district_from_label


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
