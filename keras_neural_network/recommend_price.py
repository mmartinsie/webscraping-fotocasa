"""Benchmark several network configurations and recommend a flat price.

Trains and compares several Keras configurations on the scraped Fotocasa dataset,
ranks them by how well they predict the sale price of a flat and reports the most
recommended one. The winning configuration is then retrained on the whole dataset
and used to print a recommended price.

    python recommend_price.py [dataset.csv] [--epochs N] [--drop-precio-m2]

Preprocessing:
- Features are standardized (``StandardScaler``), fit on the training set only.
- The target ``Precio`` is trained on its raw EUR scale. Every configuration also
  clips its predictions to a sane range (half the min .. 1.5x the max training
  price) so a diverging network cannot report absurd millions.

``Precio_m2`` is kept as a predictor by default to stay consistent with
``model.py`` / ``select_model.py``; it is strongly correlated with the target, so
pass ``--drop-precio-m2`` for a harder, more realistic benchmark.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Sequential

RANDOM_SEED = 42
DEFAULT_FEATURES = ["Precio_m2", "Habitaciones", "Aseos", "Superficie", "Parking", "Colegios"]
TARGET = "Precio"

DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15

# Candidate networks to test. Each one is trained from scratch and scored on the
# same held-out test set.
CONFIGURATIONS = [
    {"name": "2x6 / adam / mse",               "hidden": [6, 6],       "optimizer": "adam",    "loss": "mse"},
    {"name": "4x6 / adam / mse",               "hidden": [6, 6, 6, 6], "optimizer": "adam",    "loss": "mse"},
    {"name": "3x12 / adam / mse",              "hidden": [12, 12, 12], "optimizer": "adam",    "loss": "mse"},
    {"name": "3x24 / adam / mae",              "hidden": [24, 24, 24], "optimizer": "adam",    "loss": "mae"},
    {"name": "pyramid 32-16-8 / adam / huber", "hidden": [32, 16, 8],  "optimizer": "adam",    "loss": "huber"},
    {"name": "3x24 / rmsprop / mse",           "hidden": [24, 24, 24], "optimizer": "rmsprop", "loss": "mse"},
]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def load_dataset(csv_route: str, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(csv_route, header=0, encoding="latin1")
    data = raw.fillna(value=1)
    missing = [c for c in features + [TARGET] if c not in data.columns]
    if missing:
        raise SystemExit(f"Dataset {csv_route} is missing columns: {missing}")
    X = data[features].astype("float32").to_numpy()
    y = data[TARGET].astype("float32").to_numpy()
    return X, y


def build_model(n_features: int, hidden: list[int], optimizer: str, loss: str) -> Sequential:
    model = Sequential([Input(shape=(n_features,))])
    for units in hidden:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100),
    }


def train(
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    monitor: str = "val_loss",
) -> Sequential:
    model = build_model(X_train.shape[1], cfg["hidden"], cfg["optimizer"], cfg["loss"])
    stopper = EarlyStopping(
        monitor=monitor, patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", default="finalDataset3.csv")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--drop-precio-m2",
        action="store_true",
        help="exclude the leaky Precio_m2 feature",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    features = [f for f in DEFAULT_FEATURES if not (args.drop_precio_m2 and f == "Precio_m2")]
    set_seeds(args.seed)

    X, y = load_dataset(args.dataset, features)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=args.seed
    )

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
        print(
            "   MAE {mae:>12,.0f}   RMSE {rmse:>12,.0f}   MAPE {mape:6.1f}%   R2 {r2:7.3f}".format(
                **metrics
            )
        )

    # Lower MAE is better.
    results.sort(key=lambda item: item[1]["mae"])

    print("\n===================== Ranking (best first) =====================")
    print("{:<32} {:>13} {:>13} {:>8} {:>8}".format("configuration", "MAE", "RMSE", "MAPE", "R2"))
    for cfg, m in results:
        print(
            "{:<32} {:>13,.0f} {:>13,.0f} {:>7.1f}% {:>8.3f}".format(
                cfg["name"], m["mae"], m["rmse"], m["mape"], m["r2"]
            )
        )

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
    final_model = train(
        best_cfg, full_scaler.transform(X), y, args.epochs, args.batch_size, monitor="loss"
    )

    def recommend_price(flat: dict[str, float]) -> float:
        row = np.array([[flat[f] for f in features]], dtype="float32")
        return float(predict(final_model, full_scaler.transform(row))[0])

    # Example flat = column-wise medians of the dataset.
    sample = {f: float(np.median(X[:, i])) for i, f in enumerate(features)}
    print("\nSample flat (dataset medians):")
    for f in features:
        print(f"   {f:<12} {sample[f]:g}")
    print(f"\n>>> Recommended price: {recommend_price(sample):,.0f} EUR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
