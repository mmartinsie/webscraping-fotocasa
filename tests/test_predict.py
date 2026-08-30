import pytest

from predict import BundleError, collect_flat, load_bundle, parse_args, resolve_features

METADATA = {
    "features": ["Habitaciones", "Aseos", "Superficie"],
    "feature_medians": {"Habitaciones": 3, "Aseos": 2, "Superficie": 90},
}


def test_resolve_features_defaults_to_median():
    resolved = resolve_features(METADATA, {"Habitaciones": 5})
    assert resolved == {"Habitaciones": 5.0, "Aseos": 2.0, "Superficie": 90.0}


def test_collect_flat_set_pairs_and_json():
    args = parse_args(["dir", "--json", '{"Aseos": 1}', "--set", "Superficie=75"])
    assert collect_flat(args) == {"Aseos": 1, "Superficie": 75.0}


def test_collect_flat_rejects_bare_set():
    args = parse_args(["dir", "--set", "Superficie"])
    with pytest.raises(SystemExit):
        collect_flat(args)


def test_load_bundle_missing_dir_raises_bundleerror(tmp_path):
    with pytest.raises(BundleError):
        load_bundle(str(tmp_path))
