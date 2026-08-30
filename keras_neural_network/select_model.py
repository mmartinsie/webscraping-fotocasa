"""Compare optimizers for the price-prediction model with k-fold cross-validation.

For each candidate optimizer a fresh network is trained on every fold and scored
by mean squared error on the held-out fold; the optimizer with the lowest mean
MSE wins.

    python select_model.py --optimizers SGD RMSprop Adam --folds 3
"""

from __future__ import annotations

import argparse

import keras
import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from keras_neural_network.dataset import FEATURES, LEAKY_FEATURE, DatasetError, load_xy

RANDOM_SEED = 42
MODEL_FEATURES = [LEAKY_FEATURE, *FEATURES]


def create_model(optimizer: str) -> Sequential:
    model = Sequential(
        [
            Input(shape=(len(MODEL_FEATURES),)),
            Dense(6, activation="relu"),
            Dense(6, activation="relu"),
            Dense(1, activation="relu"),
        ]
    )
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
    return model


def cross_val_mse(
    optimizer: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: int,
    epochs: int,
    batch_size: int,
) -> np.ndarray:
    """Return the per-fold validation MSE for ``optimizer``."""
    kfold = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    scores = []
    for train_idx, val_idx in kfold.split(X):
        model = create_model(optimizer)
        model.fit(X[train_idx], y[train_idx], epochs=epochs, batch_size=batch_size, verbose=0)
        pred = model.predict(X[val_idx], verbose=0).ravel()
        scores.append(mean_squared_error(y[val_idx], pred))
    return np.array(scores)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="finalDataset3.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--optimizers", nargs="+", default=["SGD", "RMSprop", "Adam"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)

    try:
        X, y = load_xy(args.dataset, MODEL_FEATURES)
    except DatasetError as exc:
        raise SystemExit(str(exc)) from exc

    results = {}
    for optimizer in args.optimizers:
        scores = cross_val_mse(optimizer, X, y, args.folds, args.epochs, args.batch_size)
        results[optimizer] = scores
        print(f"{optimizer:<10} MSE {scores.mean():,.0f} (+/- {scores.std():,.0f})")

    best = min(results, key=lambda name: results[name].mean())
    print(f"\nBest: {best} (mean MSE {results[best].mean():,.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
