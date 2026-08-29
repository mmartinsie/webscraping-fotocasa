"""
recommend_price.py
------------------
Trains and compares several neural-network configurations ("neurons") on the
scraped Fotocasa dataset, ranks them by how well they predict the sale price of
a flat and reports the most recommended one. The winning configuration is then
retrained on the whole dataset and used to output a recommended price.

Usage:
    cd keras_neural_network
    python recommend_price.py [path/to/dataset.csv]

If no dataset path is given, ``finalDataset3.csv`` is used.

Note: ``Precio_m2`` is kept as a predictor to stay consistent with ``model.py``
and ``select_model.py``. It is strongly correlated with the target, so the
reported errors are optimistic; drop it from ``FEATURES`` for a harder, more
realistic benchmark.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.layers import Dense, Input
from keras.models import Sequential

RANDOM_SEED = 42
FEATURES = ["Precio_m2", "Habitaciones", "Aseos", "Superficie", "Parking", "Colegios"]
TARGET = "Precio"

EPOCHS = 120
BATCH_SIZE = 32

# Candidate networks to test. Each one is trained from scratch and scored on the
# same held-out test set.
CONFIGURATIONS = [
    {"name": "2x6 / adam / msle",        "hidden": [6, 6],       "optimizer": "adam",    "loss": "mean_squared_logarithmic_error"},
    {"name": "4x6 / adam / msle",        "hidden": [6, 6, 6, 6], "optimizer": "adam",    "loss": "mean_squared_logarithmic_error"},
    {"name": "4x6 / sgd / msle",         "hidden": [6, 6, 6, 6], "optimizer": "sgd",     "loss": "mean_squared_logarithmic_error"},
    {"name": "3x12 / adam / mse",        "hidden": [12, 12, 12], "optimizer": "adam",    "loss": "mean_squared_error"},
    {"name": "pyramid 16-8-4 / rmsprop", "hidden": [16, 8, 4],   "optimizer": "rmsprop", "loss": "mean_squared_error"},
    {"name": "3x24 / adam / mae",        "hidden": [24, 24, 24], "optimizer": "adam",    "loss": "mean_absolute_error"},
]


def load_dataset(csv_route):
    raw = pd.read_csv(csv_route, header=0, encoding="latin1")
    data = raw.fillna(value=1)
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise SystemExit("Dataset {} is missing columns: {}".format(csv_route, missing))
    X = data[FEATURES].astype("float32").to_numpy()
    y = data[TARGET].astype("float32").to_numpy()
    return X, y


def build_model(n_features, hidden, optimizer, loss):
    model = Sequential()
    model.add(Input(shape=(n_features,)))
    model.add(Dense(hidden[0], activation="relu"))
    for units in hidden[1:]:
        model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model


def score(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100),
    }


def evaluate_configuration(cfg, X_train, X_test, y_train, y_test):
    model = build_model(X_train.shape[1], cfg["hidden"], cfg["optimizer"], cfg["loss"])
    model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS,
              batch_size=BATCH_SIZE, verbose=0)
    y_pred = model.predict(X_test, verbose=0).ravel()
    return score(y_test, y_pred)


def main():
    csv_route = sys.argv[1] if len(sys.argv) > 1 else "finalDataset3.csv"
    if not os.path.exists(csv_route):
        raise SystemExit("Dataset not found: {}".format(csv_route))

    np.random.seed(RANDOM_SEED)
    try:
        import tensorflow as tf

        tf.random.set_seed(RANDOM_SEED)
    except Exception:
        pass

    X, y = load_dataset(csv_route)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED
    )

    # Standardize on the training set only, then reuse for the test set.
    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("Dataset: {}  |  samples: {}  |  features: {}".format(csv_route, len(X), FEATURES))
    print("Train / test split: {} / {}\n".format(len(X_train), len(X_test)))

    results = []
    for cfg in CONFIGURATIONS:
        print("Training  {} ...".format(cfg["name"]))
        metrics = evaluate_configuration(cfg, X_train, X_test, y_train, y_test)
        results.append((cfg, metrics))
        print("   MAE {mae:>12,.0f}   RMSE {rmse:>12,.0f}   MAPE {mape:6.1f}%   R2 {r2:7.3f}".format(**metrics))

    # Lower MAE is better.
    results.sort(key=lambda item: item[1]["mae"])

    print("\n===================== Ranking (best first) =====================")
    print("{:<28} {:>13} {:>13} {:>8} {:>8}".format("configuration", "MAE", "RMSE", "MAPE", "R2"))
    for cfg, m in results:
        print("{:<28} {:>13,.0f} {:>13,.0f} {:>7.1f}% {:>8.3f}".format(
            cfg["name"], m["mae"], m["rmse"], m["mape"], m["r2"]))

    best_cfg, best_metrics = results[0]
    print("\n>>> Most recommended configuration: {}".format(best_cfg["name"]))
    print("    hidden layers = {} | optimizer = {} | loss = {}".format(
        best_cfg["hidden"], best_cfg["optimizer"], best_cfg["loss"]))
    print("    expected error ~ {:,.0f} EUR (MAE) / {:.1f}% (MAPE) / R2 {:.3f}".format(
        best_metrics["mae"], best_metrics["mape"], best_metrics["r2"]))

    # Retrain the winner on every row so the final predictor uses all the data.
    print("\nRetraining the winning model on the full dataset ...")
    full_scaler = StandardScaler().fit(X)
    final_model = build_model(X.shape[1], best_cfg["hidden"], best_cfg["optimizer"], best_cfg["loss"])
    final_model.fit(full_scaler.transform(X), y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    def recommend_price(flat):
        row = np.array([[flat[f] for f in FEATURES]], dtype="float32")
        return float(final_model.predict(full_scaler.transform(row), verbose=0).ravel()[0])

    # Example flat = column-wise medians of the dataset.
    sample = {f: float(np.median(X[:, i])) for i, f in enumerate(FEATURES)}
    print("\nSample flat (dataset medians):")
    for f in FEATURES:
        print("   {:<12} {:g}".format(f, sample[f]))
    print("\n>>> Recommended price: {:,.0f} EUR".format(recommend_price(sample)))


if __name__ == "__main__":
    main()
