"""Train the thesis price-prediction model with Keras.

A ``Sequential`` network with four hidden ``Dense(6, relu)`` layers and a single
``relu`` output, trained with SGD and mean squared logarithmic error. This is the
model as used in the thesis; ``recommend_price.py`` is the tuned variant that
benchmarks several configurations and uses a linear output.

Based on https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

    python model.py --dataset finalDataset3.csv --epochs 150
"""

from __future__ import annotations

import argparse

import keras
import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential

from dataset import FEATURES, LEAKY_FEATURE, DatasetError, load_xy

RANDOM_SEED = 42

# The thesis model keeps Precio_m2 as an input (see recommend_price.py for why
# that leaks the target).
MODEL_FEATURES = [LEAKY_FEATURE, *FEATURES]

OPTIMIZERS = {
    "sgd": keras.optimizers.SGD,
    "rmsprop": keras.optimizers.RMSprop,
    "adam": keras.optimizers.Adam,
}


def build_model(optimizer: str, loss: str, learning_rate: float) -> Sequential:
    model = Sequential(
        [
            Input(shape=(len(MODEL_FEATURES),)),
            Dense(6, activation="relu"),
            Dense(6, activation="relu"),
            Dense(6, activation="relu"),
            Dense(6, activation="relu"),
            Dense(1, activation="relu"),
        ]
    )
    model.compile(
        optimizer=OPTIMIZERS[optimizer](learning_rate=learning_rate),
        loss=loss,
        metrics=["msle"],
    )
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="finalDataset3.csv")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--optimizer", choices=sorted(OPTIMIZERS), default="sgd")
    parser.add_argument("--loss", default="mean_squared_logarithmic_error")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--validation-split", type=float, default=0.30)
    parser.add_argument("--plot", action="store_true", help="plot the training history")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)

    try:
        X, y = load_xy(args.dataset, MODEL_FEATURES)
    except DatasetError as exc:
        raise SystemExit(str(exc)) from exc
    model = build_model(args.optimizer, args.loss, args.learning_rate)
    history = model.fit(
        X,
        y,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("Minimum training MSLE:", float(np.min(history.history["msle"])))

    if args.plot:
        import matplotlib.pyplot as plt

        plt.plot(history.history["msle"], label="train")
        if "val_msle" in history.history:
            plt.plot(history.history["val_msle"], label="validation")
        plt.xlabel("epoch")
        plt.ylabel("MSLE")
        plt.legend()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
