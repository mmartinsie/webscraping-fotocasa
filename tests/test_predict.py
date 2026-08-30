import pytest

from predict import BundleError, collect_flat, feature_vector, load_bundle, parse_args, resolve_numeric

METADATA = {
    "numeric_features": ["Habitaciones", "Aseos", "Superficie"],
    "district_categories": ["Centro", "Retiro", "Salamanca"],
    "feature_medians": {"Habitaciones": 3, "Aseos": 2, "Superficie": 90},
}


def test_resolve_numeric_defaults_to_median():
    resolved = resolve_numeric(METADATA, {"Habitaciones": 5})
    assert resolved == {"Habitaciones": 5.0, "Aseos": 2.0, "Superficie": 90.0}


def test_feature_vector_appends_district_one_hot():
    vec = feature_vector(METADATA, {"Superficie": 100, "Distrito": "Retiro"})
    assert vec == [3.0, 2.0, 100.0, 0.0, 1.0, 0.0]  # medians + one-hot(Retiro)


def test_feature_vector_unknown_district_is_all_zeros():
    vec = feature_vector(METADATA, {"Distrito": "Narnia"})
    assert vec[-3:] == [0.0, 0.0, 0.0]


def test_collect_flat_keeps_strings_and_floats():
    args = parse_args(["dir", "--json", '{"Aseos": 1}', "--set", "Superficie=75", "--set", "Distrito=Retiro"])
    assert collect_flat(args) == {"Aseos": 1, "Superficie": 75.0, "Distrito": "Retiro"}


def test_collect_flat_rejects_bare_set():
    args = parse_args(["dir", "--set", "Superficie"])
    with pytest.raises(SystemExit):
        collect_flat(args)


def test_load_bundle_missing_dir_raises_bundleerror(tmp_path):
    with pytest.raises(BundleError):
        load_bundle(str(tmp_path))
