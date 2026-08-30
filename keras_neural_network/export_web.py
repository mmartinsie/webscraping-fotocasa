"""Export the GitHub Pages demo data.

Writes ``docs/model.json`` (the thesis network's weights + scaler + band, so the
page can run the forward pass in ~15 lines of JavaScript instead of shipping the
TensorFlow.js runtime) and ``docs/districts.json`` (the €/m² reference table).

    python recommend_price.py --save web_model
    python export_web.py web_model -o ../docs/model.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os

from keras_neural_network.predict import load_bundle, predict_price

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DISTRICT_CSV = os.path.join(_DATA, "precio_m2_distrito.csv")
SCHOOLS_CSV = os.path.join(_DATA, "colegios_distrito.csv")


def _read(path: str, value_col: str) -> dict:
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["Distrito"]: float(row[value_col]) for row in csv.DictReader(handle)}


def write_districts(output_dir: str) -> None:
    eur_m2 = _read(DISTRICT_CSV, "EurM2")
    schools = _read(SCHOOLS_CSV, "Colegios")
    table = {name: {"eur_m2": eur_m2[name], "colegios": schools.get(name)} for name in eur_m2}
    path = os.path.join(output_dir, "districts.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(table, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {path} ({len(table)} districts)")


def export(model_dir: str, output: str) -> None:
    model, scaler, metadata = load_bundle(model_dir)

    layers = []
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:  # e.g. the Input layer
            continue
        kernel, bias = weights
        layers.append(
            {
                "W": kernel.tolist(),  # shape (n_in, n_out)
                "b": bias.tolist(),  # shape (n_out,)
                "activation": layer.get_config().get("activation", "linear"),
            }
        )

    band = metadata.get("price_band", {})
    cfg = metadata.get("configuration", {})
    m = metadata.get("cv_metrics") or metadata.get("test_metrics", {})
    numeric = metadata.get("numeric_features") or metadata.get("features", [])
    categories = metadata.get("district_categories", [])
    bundle = {
        "numeric_features": numeric,
        "district_categories": categories,
        "feature_medians": metadata.get("feature_medians", {}),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "band": {"low": band.get("low"), "high": band.get("high")},
        "layers": layers,
        "meta": {
            "config": cfg.get("name"),
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
            "r2": m.get("r2"),
            "mape": m.get("mape"),
            "with_district": bool(categories),
            "generated": dt.date.today().isoformat(),
        },
        # Regression anchor: the Keras prediction for the all-median flat with no
        # district. The pure-JS / NumPy forward passes must reproduce it.
        "_golden_median_price": round(predict_price(model, scaler, metadata, {}), 2),
    }

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
    print(f"Wrote {output} ({len(layers)} layers, {len(numeric)} numeric + {len(categories)} districts)")
    write_districts(out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir", help="bundle dir from recommend_price.py --save")
    parser.add_argument("-o", "--output", default="../docs/model.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.isdir(args.model_dir):
        raise SystemExit(f"No model bundle at {args.model_dir!r}")
    export(args.model_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
