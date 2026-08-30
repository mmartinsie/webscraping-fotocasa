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

from predict import load_bundle, predict_price

DISTRICT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "precio_m2_distrito.csv")


def write_districts(output_dir: str) -> None:
    with open(DISTRICT_CSV, encoding="utf-8", newline="") as handle:
        table = {row["Distrito"]: float(row["EurM2"]) for row in csv.DictReader(handle)}
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
    bundle = {
        "features": metadata["features"],
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
            "generated": dt.date.today().isoformat(),
        },
        # Regression anchor: the Keras prediction for the all-median flat. The
        # pure-JS / NumPy forward passes must reproduce it (see tests/test_pricing).
        "_golden_median_price": round(predict_price(model, scaler, metadata, {}), 2),
    }

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
    print(f"Wrote {output} ({len(layers)} dense layers, {len(bundle['features'])} features)")
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
