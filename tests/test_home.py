from webscraping.home import CSV_HEADERS, Home


def test_to_csv_row_keys_match_headers():
    row = Home(url="http://x", district="Salamanca").to_csv_row()
    assert list(row) == CSV_HEADERS


def test_to_csv_row_values():
    home = Home(
        url="http://x",
        district="Centro",
        price=250_000,
        property_type="Piso",
        rooms=3,
        baths=2,
        size=90,
        floor=4,
        parking=True,
    )
    row = home.to_csv_row()
    assert row["Precio"] == 250_000
    assert row["Distrito"] == "Centro"
    assert row["Tipo"] == "Piso"
    assert row["Parking"] == 1
    assert row["URL"] == "http://x"


def test_parking_false_is_blank():
    assert Home(url="u", district="d").to_csv_row()["Parking"] is None
