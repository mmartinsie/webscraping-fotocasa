import pandas as pd
import pytest

from dataset import FEATURES, DatasetError, feature_medians, load_xy, read_csv


def _write(tmp_path, rows, encoding="utf-8"):
    path = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding=encoding)
    return str(path)


def test_load_xy_fills_missing_with_median(tmp_path):
    path = _write(
        tmp_path,
        {
            "Habitaciones": [2, 4, None],
            "Aseos": [1, 2, 2],
            "Superficie": [60, 120, 90],
            "Parking": [0, 1, 0],
            "Colegios": [5, 9, 7],
            "Precio": [100000, 300000, 200000],
        },
    )
    X, y = load_xy(path, FEATURES)
    assert X.shape == (3, 5)
    # median of [2, 4] is 3.0
    assert X[2, 0] == pytest.approx(3.0)
    assert list(y) == [100000, 300000, 200000]


def test_load_xy_missing_column_raises_dataseterror(tmp_path):
    path = _write(tmp_path, {"Habitaciones": [1], "Precio": [1]})
    with pytest.raises(DatasetError):
        load_xy(path, FEATURES)


def test_read_csv_handles_latin1(tmp_path):
    path = tmp_path / "legacy.csv"
    path.write_bytes("Distrito,Precio\nChamart\xedn,100000\n".encode("latin1"))
    df = read_csv(str(path))
    assert df.loc[0, "Distrito"] == "Chamartín"


def test_feature_medians(tmp_path):
    path = _write(
        tmp_path,
        {
            "Habitaciones": [1, 3, 5],
            "Aseos": [1, 1, 1],
            "Superficie": [1, 1, 1],
            "Parking": [0, 0, 0],
            "Colegios": [0, 0, 0],
            "Precio": [1, 1, 1],
        },
    )
    med = feature_medians(path, FEATURES)
    assert med["Habitaciones"] == 3.0
    assert set(med) == set(FEATURES)
