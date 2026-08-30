"""Non-neural baselines for the price-prediction task.

Trains a mean predictor, a linear regression and a random forest on the same
features / split as ``recommend_price.py`` so the network has something to beat.
``--with-district`` one-hot-encodes ``Distrito`` to show how much location alone
is worth.

    python baseline.py [dataset.csv] [--with-district]
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from keras_neural_network.dataset import FEATURES, LEAKY_FEATURE, TARGET, read_csv
from keras_neural_network.metrics import format_row, score

RANDOM_SEED = 42


def load_features(dataset: str, features: list[str], with_district: bool):
    """Return ``(X, y)`` with median-filled numeric features, plus optional
    one-hot ``Distrito`` columns."""
    raw = read_csv(dataset)
    numeric = raw[[*features, TARGET]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    X = numeric[features].reset_index(drop=True)
    if with_district and "Distrito" in raw.columns:
        dummies = pd.get_dummies(raw["Distrito"], prefix="Distrito").reset_index(drop=True)
        X = pd.concat([X, dummies], axis=1)
    return X.to_numpy(dtype="float32"), numeric[TARGET].to_numpy(dtype="float32"), list(X.columns)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", default="finalDataset3.csv")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--keep-precio-m2", action="store_true", help="keep the leaky Precio_m2 feature")
    parser.add_argument("--with-district", action="store_true", help="add one-hot Distrito columns")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.dataset):
        raise SystemExit(f"Dataset not found: {args.dataset}")

    features = ([LEAKY_FEATURE] if args.keep_precio_m2 else []) + FEATURES
    X, y, columns = load_features(args.dataset, features, args.with_district)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=args.seed)

    models = {
        "mean (DummyRegressor)": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=args.seed, n_jobs=-1),
    }

    print(f"Dataset: {args.dataset}  |  {len(columns)} columns  |  district={args.with_district}")
    print(f"Train / test split: {len(X_train)} / {len(X_test)}\n")
    print(f"{'model':<24} {'MAE':>13} {'RMSE':>13} {'MAPE':>8} {'R2':>8}")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(format_row(name, score(y_test, model.predict(X_test)), width=24))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
