"""Benchmark several network configurations and recommend a flat price.

Trains and compares several Keras configurations on the scraped Fotocasa dataset,
ranks them by how well they predict the sale price of a flat and reports the most
recommended one. The winning configuration is then retrained on the whole dataset
and used to print a recommended price; ``--save DIR`` persists it for
``predict.py``.

    python recommend_price.py [dataset.csv] [--epochs N] [--save model_dir]

Preprocessing:
- Features are standardized (``StandardScaler``), fit on the training set only.
- The target ``Precio`` is trained on its raw EUR scale. Every configuration also
  clips its predictions to a sane range (half the min .. 1.5x the max training
  price) so a diverging network cannot report absurd millions.

By default ``Precio_m2`` is excluded: ``precio == precio_m2 * superficie`` almost
exactly, so keeping it leaks the target and inflates the scores. Pass
``--keep-precio-m2`` to reproduce the leaky setup used by ``model.py``.
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import FEATURES as DEFAULT_FEATURES
from dataset import LEAKY_FEATURE, TARGET, load_xy
from metrics import format_row, score

RANDOM_SEED = 42

DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15

# Candidate networks to test. Each one is trained from scratch and scored on the
# same held-out test set.
CONFIGURATIONS = [
    {"name": "2x6 / adam / mse", "hidden": [6, 6], "optimizer": "adam", "loss": "mse"},
    {"name": "4x6 / adam / mse", "hidden": [6, 6, 6, 6], "optimizer": "adam", "loss": "mse"},
    {"name": "3x12 / adam / mse", "hidden": [12, 12, 12], "optimizer": "adam", "loss": "mse"},
    {"name": "3x24 / adam / mae", "hidden": [24, 24, 24], "optimizer": "adam", "loss": "mae"},
    {"name": "pyramid 32-16-8 / adam / huber", "hidden": [32, 16, 8], "optimizer": "adam", "loss": "huber"},
    {"name": "3x24 / rmsprop / mse", "hidden": [24, 24, 24], "optimizer": "rmsprop", "loss": "mse"},
]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(n_features: int, hidden: list[int], optimizer: str, loss: str) -> Sequential:
    model = Sequential([Input(shape=(n_features,))])
    for units in hidden:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model


def train(
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    monitor: str = "val_loss",
) -> Sequential:
    model = build_model(X_train.shape[1], cfg["hidden"], cfg["optimizer"], cfg["loss"])
    stopper = EarlyStopping(monitor=monitor, patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    validation_split = 0.2 if monitor.startswith("val") else 0.0
    model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[stopper],
    )
    return model


def save_bundle(
    directory: str,
    model: Sequential,
    scaler: StandardScaler,
    features: list[str],
    band: tuple[float, float],
    cfg: dict,
    metrics: dict[str, float],
) -> None:
    """Persist everything ``predict.py`` needs into ``directory``."""
    os.makedirs(directory, exist_ok=True)
    model.save(os.path.join(directory, "model.keras"))
    joblib.dump(scaler, os.path.join(directory, "scaler.joblib"))
    metadata = {
        "features": features,
        "target": TARGET,
        "price_band": {"low": band[0], "high": band[1]},
        "configuration": cfg,
        "test_metrics": metrics,
        "seed": RANDOM_SEED,
    }
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nSaved model bundle to {directory}/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", default="finalDataset3.csv")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--keep-precio-m2",
        action="store_true",
        help="keep the leaky Precio_m2 feature (reproduces model.py's setup)",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        help="save the recommended model, scaler and metadata to DIR",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    features = ([LEAKY_FEATURE] if args.keep_precio_m2 else []) + DEFAULT_FEATURES
    set_seeds(args.seed)

    X, y = load_xy(args.dataset, features)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=args.seed)

    # Standardize features on the training set only, then reuse for the test set.
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Guardrail: keep predictions inside a sane price band.
    lo, hi = float(y_train.min()) * 0.5, float(y_train.max()) * 1.5

    def predict(model: Sequential, feats: np.ndarray) -> np.ndarray:
        return np.clip(model.predict(feats, verbose=0).ravel(), lo, hi)

    print(f"Dataset: {args.dataset}  |  samples: {len(X)}  |  features: {features}")
    print(f"Train / test split: {len(X_train)} / {len(X_test)}\n")

    results = []
    for cfg in CONFIGURATIONS:
        print(f"Training  {cfg['name']} ...")
        model = train(cfg, X_train, y_train, args.epochs, args.batch_size)
        metrics = score(y_test, predict(model, X_test))
        results.append((cfg, metrics))
        print("   " + format_row(cfg["name"], metrics))

    # Lower MAE is better.
    results.sort(key=lambda item: item[1]["mae"])

    print("\n===================== Ranking (best first) =====================")
    print(f"{'configuration':<32} {'MAE':>13} {'RMSE':>13} {'MAPE':>8} {'R2':>8}")
    for cfg, m in results:
        print(format_row(cfg["name"], m))

    best_cfg, best_metrics = results[0]
    print(f"\n>>> Most recommended configuration: {best_cfg['name']}")
    print(
        "    hidden layers = {} | optimizer = {} | loss = {}".format(
            best_cfg["hidden"], best_cfg["optimizer"], best_cfg["loss"]
        )
    )
    print(
        "    expected error ~ {:,.0f} EUR (MAE) / {:.1f}% (MAPE) / R2 {:.3f}".format(
            best_metrics["mae"], best_metrics["mape"], best_metrics["r2"]
        )
    )

    # Retrain the winner on every row so the final predictor uses all the data.
    print("\nRetraining the winning model on the full dataset ...")
    full_scaler = StandardScaler().fit(X)
    final_model = train(best_cfg, full_scaler.transform(X), y, args.epochs, args.batch_size, monitor="loss")

    def recommend_price(flat: dict[str, float]) -> float:
        row = np.array([[flat[f] for f in features]], dtype="float32")
        return float(predict(final_model, full_scaler.transform(row))[0])

    # Example flat = column-wise medians of the dataset.
    sample = {f: float(np.median(X[:, i])) for i, f in enumerate(features)}
    print("\nSample flat (dataset medians):")
    for f in features:
        print(f"   {f:<12} {sample[f]:g}")
    print(f"\n>>> Recommended price: {recommend_price(sample):,.0f} EUR")

    if args.save:
        save_bundle(args.save, final_model, full_scaler, features, (lo, hi), best_cfg, best_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
