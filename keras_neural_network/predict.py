"""Predict a flat's price from a saved model bundle.

Uses a directory produced by ``recommend_price.py --save DIR`` (``model.keras``,
``scaler.joblib``, ``metadata.json``).

    python predict.py model_dir --Habitaciones 3 --Aseos 2 --Superficie 90 \
        --Parking 0 --Colegios 9
    python predict.py model_dir --json '{"Habitaciones": 3, "Superficie": 90}'

Any feature not given defaults to 0.
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import keras
import numpy as np


def load_bundle(directory: str):
    with open(os.path.join(directory, "metadata.json"), encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = keras.models.load_model(os.path.join(directory, "model.keras"))
    scaler = joblib.load(os.path.join(directory, "scaler.joblib"))
    return model, scaler, metadata


def predict_price(model, scaler, metadata: dict, flat: dict[str, float]) -> float:
    features = metadata["features"]
    row = np.array([[float(flat.get(name, 0.0)) for name in features]], dtype="float32")
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
        "--feature",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="set one feature (repeatable), e.g. --feature Superficie=90",
    )
    # Parse the rest as --<FeatureName> VALUE too, resolved after we know the names.
    args, extra = parser.parse_known_args(argv)
    args.extra = extra
    return args


def collect_flat(args: argparse.Namespace, feature_names: list[str]) -> dict[str, float]:
    flat: dict[str, float] = {}
    if args.json:
        flat.update(json.loads(args.json))
    for item in args.feature:
        name, _, value = item.partition("=")
        flat[name] = float(value)
    # --<FeatureName> VALUE pairs
    tokens = list(args.extra)
    while tokens:
        token = tokens.pop(0)
        if token.startswith("--") and token[2:] in feature_names and tokens:
            flat[token[2:]] = float(tokens.pop(0))
    return flat


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, scaler, metadata = load_bundle(args.model_dir)
    flat = collect_flat(args, metadata["features"])

    price = predict_price(model, scaler, metadata, flat)
    used = {name: flat.get(name, 0.0) for name in metadata["features"]}
    print(f"Features: {used}")
    print(f"Predicted price: {price:,.0f} EUR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
