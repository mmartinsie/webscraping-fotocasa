import pandas as pd

from prepare_dataset import OUTPUT_COLUMNS, attach_schools, clean

RAW = pd.DataFrame(
    {
        "Precio": [200000, 5000, 300000, 300000],
        "Distrito": ["Centro", "Centro", "Retiro", "Retiro"],
        "Tipo": ["Piso", "Piso", "Piso", "Piso"],
        "Habitaciones": [2, 1, 3, 3],
        "Aseos": [1, 1, 2, 2],
        "Superficie": [80, 40, 100, 100],
        "Planta": [3, 1, 5, 5],
        "Parking": [None, None, 1, 1],
        "URL": ["a", "b", "c", "c"],
    }
)


def _clean(df):
    return clean(df, min_price=20_000, max_price=3_000_000, min_size=20, max_size=1_000)


def test_clean_drops_out_of_band_and_duplicates():
    out = _clean(RAW)
    # 5000 EUR row dropped (below min_price), duplicate Retiro row collapsed.
    assert len(out) == 2
    assert set(out["Precio"]) == {200000, 300000}


def test_clean_computes_precio_m2_and_parking():
    out = _clean(RAW).reset_index(drop=True)
    assert out.loc[out["Precio"] == 200000, "Precio_m2"].iloc[0] == 2500
    assert out.loc[out["Precio"] == 200000, "Parking"].iloc[0] == 0
    assert out.loc[out["Precio"] == 300000, "Parking"].iloc[0] == 1


def test_attach_schools_without_csv_leaves_column_empty():
    out = attach_schools(_clean(RAW), None).reindex(columns=OUTPUT_COLUMNS)
    assert out["Colegios"].isna().all()
    assert list(out.columns) == OUTPUT_COLUMNS


def test_attach_schools_joins_on_district(tmp_path):
    schools = tmp_path / "schools.csv"
    schools.write_text("Distrito,Colegios\nCentro,7\nRetiro,4\n")
    out = attach_schools(_clean(RAW), str(schools))
    assert set(out["Colegios"]) == {7, 4}
