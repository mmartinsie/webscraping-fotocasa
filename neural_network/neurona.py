"""Train the from-scratch network on the scraped dataset.

OBSOLETE first iteration, kept for reference. The maintained model lives in
``../keras_neural_network``. Based on
https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/

    python neurona.py --dataset ../keras_neural_network/buildings_information.csv
"""

import argparse
import sys

import numpy as np
import pandas

from capas import capa
from entrenamiento import entrenamiento, mse
from funcionRelu import relu

np.set_printoptions(threshold=sys.maxsize)

# Layer sizes: 1 input feature (rooms) -> hidden layer of 2 -> 1 output (price).
NEURONAS = [1, 2, 1]
EPOCHS = 2
LEARNING_RATE = 0.001


def load_xy(dataset):
    """Return ``(X, Y)`` with X = number of rooms, Y = price."""
    raw = pandas.read_csv(dataset, header=0, encoding="latin1")
    data = raw.fillna(value=1).drop(["Distrito", "Tipo", "URL"], axis=1)
    y = data["Precio"].to_numpy().reshape(-1, 1)
    x = data["Habitaciones"].to_numpy().reshape(-1, 1)
    return x, y


def build_network():
    """Build the list of layers described by :data:`NEURONAS`."""
    return [capa(NEURONAS[i], NEURONAS[i + 1], relu) for i in range(len(NEURONAS) - 1)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        default="buildings_information.csv",
        help="CSV produced by the scraper (default: buildings_information.csv)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--plot", action="store_true", help="plot the training error")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    X, Y = load_xy(args.dataset)
    network = build_network()

    errors = []
    for _ in range(args.epochs):
        prediction = entrenamiento(X=X, Y=Y, red_neuronal=network, lr=args.learning_rate)
        errors.append(mse(np.round(prediction), Y)[0])

    if args.plot:
        import matplotlib.pyplot as plt

        plt.plot(range(len(errors)), errors)
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.show()

    # Final forward pass.
    output = X
    for layer in network:
        output = layer.funcion_act[0](output @ layer.W + layer.b)

    print("Training MSE per epoch:", errors)
    print("Final predictions:", output.ravel())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
