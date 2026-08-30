"""Predict a flat's price from a saved model bundle.

Uses a directory produced by ``recommend_price.py --save DIR`` (``model.keras``,
``scaler.joblib``, ``metadata.json``).

    python predict.py model_dir --set Superficie=90 --set Habitaciones=3 --set Distrito=Salamanca
    python predict.py model_dir --json '{"Superficie": 90, "Distrito": "Retiro"}'

Numeric features not provided default to the training median in the bundle. If
the model was trained ``--with-district``, ``Distrito`` is one-hot encoded from
``metadata["district_categories"]`` (an unknown/missing district -> all zeros).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


class BundleError(Exception):
    """Raised when a model bundle directory is missing or incomplete."""


def load_bundle(directory: str):
    for name in ("model.keras", "scaler.joblib", "metadata.json"):
        if not os.path.exists(os.path.join(directory, name)):
            raise BundleError(f"{directory!r} is not a model bundle (missing {name})")

    import joblib  # heavy imports kept out of the module import path
    import keras

    with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = keras.models.load_model(os.path.join(directory, "model.keras"))
    scaler = joblib.load(os.path.join(directory, "scaler.joblib"))
    return model, scaler, metadata


def numeric_features(metadata: dict) -> list[str]:
    return metadata.get("numeric_features") or metadata.get("features", [])


def resolve_numeric(metadata: dict, given: dict) -> dict[str, float]:
    """Every numeric feature, defaulting missing ones to the stored median."""
    medians = metadata.get("feature_medians", {})
    return {name: float(given.get(name, medians.get(name, 0.0))) for name in numeric_features(metadata)}


def feature_vector(metadata: dict, flat: dict) -> list[float]:
    """Model input: numeric features then the district one-hot (if any)."""
    row = list(resolve_numeric(metadata, flat).values())
    row += [1.0 if flat.get("Distrito") == c else 0.0 for c in metadata.get("district_categories", [])]
    return row


def predict_price(model, scaler, metadata: dict, flat: dict) -> float:
    row = np.array([feature_vector(metadata, flat)], dtype="float32")
    raw = float(model.predict(scaler.transform(row), verbose=0).ravel()[0])
    band = metadata.get("price_band")
    if band:
        raw = min(max(raw, band["low"]), band["high"])
    return raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("model_dir", help="directory from recommend_price.py --save")
    parser.add_argument("--json", help="feature values as a JSON object")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        dest="pairs",
        help="set one feature (repeatable), e.g. --set Superficie=90 or --set Distrito=Retiro",
    )
    return parser.parse_args(argv)


def collect_flat(args: argparse.Namespace) -> dict:
    flat: dict = {}
    if args.json:
        flat.update(json.loads(args.json))
    for item in args.pairs:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        try:
            flat[name] = float(value)
        except ValueError:
            flat[name] = value  # e.g. Distrito=Salamanca
    return flat


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model, scaler, metadata = load_bundle(args.model_dir)
    except BundleError as exc:
        raise SystemExit(str(exc)) from exc

    flat = collect_flat(args)
    used = resolve_numeric(metadata, flat)
    if metadata.get("district_categories"):
        used["Distrito"] = flat.get("Distrito", "(none)")
    price = predict_price(model, scaler, metadata, flat)
    print(f"Features: {used}")
    print(f"Predicted price: {price:,.0f} EUR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
