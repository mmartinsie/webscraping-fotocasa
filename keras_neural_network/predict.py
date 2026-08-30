"""Predict a flat's price from a saved model bundle.

Uses a directory produced by ``recommend_price.py --save DIR`` (``model.keras``,
``scaler.joblib``, ``metadata.json``).

    python predict.py model_dir --set Habitaciones=3 --set Superficie=90
    python predict.py model_dir --json '{"Habitaciones": 3, "Superficie": 90}'

Features not provided default to the training median stored in the bundle.
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


def resolve_features(metadata: dict, given: dict[str, float]) -> dict[str, float]:
    """Fill in every model feature, defaulting missing ones to the stored median."""
    medians = metadata.get("feature_medians", {})
    return {name: float(given.get(name, medians.get(name, 0.0))) for name in metadata["features"]}


def predict_price(model, scaler, metadata: dict, flat: dict[str, float]) -> float:
    resolved = resolve_features(metadata, flat)
    row = np.array([[resolved[name] for name in metadata["features"]]], dtype="float32")
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
        help="set one feature (repeatable), e.g. --set Superficie=90",
    )
    return parser.parse_args(argv)


def collect_flat(args: argparse.Namespace) -> dict[str, float]:
    flat: dict[str, float] = {}
    if args.json:
        flat.update(json.loads(args.json))
    for item in args.pairs:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        flat[name] = float(value)
    return flat


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model, scaler, metadata = load_bundle(args.model_dir)
    except BundleError as exc:
        raise SystemExit(str(exc)) from exc

    flat = collect_flat(args)
    used = resolve_features(metadata, flat)
    price = predict_price(model, scaler, metadata, flat)
    print(f"Features: {used}")
    print(f"Predicted price: {price:,.0f} EUR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
