# Neural network (from scratch) — deprecated

> ⚠️ **Obsolete.** This was the first iteration of the price-prediction model,
> written from scratch with NumPy. It is kept for reference only. The working
> implementation lives in [`../keras_neural_network`](../keras_neural_network).

A minimal feed-forward neural network implemented without any deep-learning
framework, following
[Ander Fernández's tutorial](https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/).

## Scripts

| File | Purpose |
| --- | --- |
| `neurona.py` | Entry point. Loads `buildings_information.csv`, builds the network (layer sizes `[1, 2, 1]`, ReLU activations), trains it for a few epochs and plots the error. |
| `capas.py` | `capa` class — one dense layer. Initializes the weight matrix `W` and bias vector `b` with a truncated normal distribution. |
| `entrenamiento.py` | `entrenamiento()` (forward pass + backpropagation + gradient descent) and `mse()` (mean squared error and its derivative). |
| `funcionRelu.py` | Activation functions and their derivatives: `relu`, `idem` (identity), `sigmoid`. |

## Requirements

- Python 3.8
- Python packages:

  ```bash
  pip install numpy scipy pandas matplotlib
  ```

## Usage

```bash
cd neural_network
python neurona.py
```

Expects a `buildings_information.csv` file (produced by the
[`../webscraping`](../webscraping) scripts) in the current directory.

## Known limitations

- Only a single input feature (`Habitaciones`) is wired up.
- Weight initialization is scaled by large constants, so training is unstable.
- No train/test split, normalization or evaluation metrics.

These are the reasons the project moved to the Keras implementation.
