"""Benchmark several network configurations and recommend a flat price.

Compares several Keras configurations with k-fold cross-validation, ranks them by
mean MAE and reports the best. The winner is then retrained on the whole dataset
and used to print a recommended price; ``--save DIR`` persists it for
``predict.py``.

    python recommend_price.py [dataset.csv] [--folds 3] [--epochs N] [--save model_dir]

Preprocessing:
- Features are standardized (``StandardScaler``), fit on the training fold only.
- Missing values are filled with the column median (see ``dataset.load_xy``).
- The target ``Precio`` is trained on its raw EUR scale. Predictions are clipped
  to a sane range (half the min .. 1.5x the max training price) so a diverging
  network cannot report absurd millions.

By default ``Precio_m2`` is excluded: ``precio == precio_m2 * superficie`` almost
exactly, so keeping it leaks the target and inflates the scores. Pass
``--keep-precio-m2`` to reproduce the leaky setup used by ``model.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import NamedTuple

import joblib
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from dataset import FEATURES as DEFAULT_FEATURES
from dataset import LEAKY_FEATURE, TARGET, DatasetError, feature_medians, load_xy
from metrics import format_row, score

RANDOM_SEED = 42
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_FOLDS = 3
EARLY_STOPPING_PATIENCE = 15
VALIDATION_SPLIT = 0.2


class Config(NamedTuple):
    name: str
    hidden: list[int]
    optimizer: str
    loss: str


CONFIGURATIONS = [
    Config("2x6 / adam / mse", [6, 6], "adam", "mse"),
    Config("4x6 / adam / mse", [6, 6, 6, 6], "adam", "mse"),
    Config("3x12 / adam / mse", [12, 12, 12], "adam", "mse"),
    Config("3x24 / adam / mae", [24, 24, 24], "adam", "mae"),
    Config("pyramid 32-16-8 / adam / huber", [32, 16, 8], "adam", "huber"),
    Config("3x24 / rmsprop / mse", [24, 24, 24], "rmsprop", "mse"),
]


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(n_features: int, cfg: Config) -> Sequential:
    model = Sequential([Input(shape=(n_features,))])
    for units in cfg.hidden:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=cfg.optimizer, loss=cfg.loss)
    return model


def train(cfg: Config, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> Sequential:
    """Fit a fresh model with an inner validation split and early stopping."""
    model = build_model(x.shape[1], cfg)
    stopper = EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)
    model.fit(
        x,
        y,
        validation_split=VALIDATION_SPLIT,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[stopper],
    )
    return model


def _band(y: np.ndarray) -> tuple[float, float]:
    return float(y.min()) * 0.5, float(y.max()) * 1.5


def cross_validate(
    cfg: Config, X: np.ndarray, y: np.ndarray, folds: int, epochs: int, batch_size: int, seed: int
) -> dict[str, float]:
    """Mean MAE/RMSE/R2/MAPE of ``cfg`` over a ``folds``-way CV."""
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_scores = []
    for train_idx, test_idx in kfold.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        model = train(cfg, scaler.transform(X[train_idx]), y[train_idx], epochs, batch_size)
        lo, hi = _band(y[train_idx])
        pred = np.clip(model.predict(scaler.transform(X[test_idx]), verbose=0).ravel(), lo, hi)
        fold_scores.append(score(y[test_idx], pred))
    return {k: float(np.mean([s[k] for s in fold_scores])) for k in fold_scores[0]}


def save_bundle(
    directory: str,
    model: Sequential,
    scaler: StandardScaler,
    features: list[str],
    band: tuple[float, float],
    cfg: Config,
    cv_metrics: dict[str, float],
    medians: dict[str, float],
) -> None:
    """Persist everything ``predict.py`` needs into ``directory``."""
    os.makedirs(directory, exist_ok=True)
    model.save(os.path.join(directory, "model.keras"))
    joblib.dump(scaler, os.path.join(directory, "scaler.joblib"))
    metadata = {
        "features": features,
        "target": TARGET,
        "feature_medians": medians,
        "price_band": {"low": band[0], "high": band[1]},
        "configuration": cfg._asdict(),
        "cv_metrics": cv_metrics,
        "seed": RANDOM_SEED,
    }
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nSaved model bundle to {directory}/")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", default="finalDataset3.csv")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--keep-precio-m2",
        action="store_true",
        help="keep the leaky Precio_m2 feature (reproduces model.py's setup)",
    )
    parser.add_argument(
        "--save", metavar="DIR", help="save the recommended model, scaler and metadata to DIR"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    features = ([LEAKY_FEATURE] if args.keep_precio_m2 else []) + DEFAULT_FEATURES
    set_seeds(args.seed)
    try:
        X, y = load_xy(args.dataset, features)
    except DatasetError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Dataset: {args.dataset}  |  samples: {len(X)}  |  features: {features}")
    print(f"{args.folds}-fold CV, up to {args.epochs} epochs per fold\n")

    results = []
    for cfg in CONFIGURATIONS:
        print(f"Cross-validating  {cfg.name} ...")
        metrics = cross_validate(cfg, X, y, args.folds, args.epochs, args.batch_size, args.seed)
        results.append((cfg, metrics))
        print("   " + format_row(cfg.name, metrics))

    results.sort(key=lambda item: item[1]["mae"])

    print("\n============== CV ranking (best mean MAE first) ==============")
    print(f"{'configuration':<32} {'MAE':>13} {'RMSE':>13} {'MAPE':>8} {'R2':>8}")
    for cfg, m in results:
        print(format_row(cfg.name, m))

    best_cfg, best_metrics = results[0]
    print(f"\n>>> Most recommended configuration: {best_cfg.name}")
    print(f"    hidden = {best_cfg.hidden} | optimizer = {best_cfg.optimizer} | loss = {best_cfg.loss}")
    print(
        "    CV error ~ {:,.0f} EUR (MAE) / {:.1f}% (MAPE) / R2 {:.3f}".format(
            best_metrics["mae"], best_metrics["mape"], best_metrics["r2"]
        )
    )

    print("\nRetraining the winner on the full dataset ...")
    full_scaler = StandardScaler().fit(X)
    final_model = train(best_cfg, full_scaler.transform(X), y, args.epochs, args.batch_size)
    lo, hi = _band(y)

    def recommend_price(flat: dict[str, float]) -> float:
        row = np.array([[flat[f] for f in features]], dtype="float32")
        pred = final_model.predict(full_scaler.transform(row), verbose=0).ravel()[0]
        return float(np.clip(pred, lo, hi))

    sample = {f: float(np.median(X[:, i])) for i, f in enumerate(features)}
    print("\nSample flat (dataset medians):")
    for f in features:
        print(f"   {f:<12} {sample[f]:g}")
    print(f"\n>>> Recommended price: {recommend_price(sample):,.0f} EUR")

    if args.save:
        medians = feature_medians(args.dataset, features)
        save_bundle(args.save, final_model, full_scaler, features, (lo, hi), best_cfg, best_metrics, medians)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
