# Keras neural network

Final implementation of the model that predicts the sale price of a Madrid
property from its characteristics, built with [Keras](https://keras.io/).

See [`DATA.md`](DATA.md) for the datasets and their columns.

## Contents

| File | Purpose |
| --- | --- |
| `model.py` | Trains the thesis model (CLI): a `Sequential` network with four hidden `Dense(6, relu)` layers and a `Dense(1, relu)` output, SGD + `mean_squared_logarithmic_error`, 150 epochs, 30% validation split. Prints the minimum MSLE; `--plot` charts the history. |
| `select_model.py` | Compares optimizers (`SGD`, `RMSprop`, `Adam` by default) with k-fold cross-validation, scoring by mean squared error, and reports the best. |
| `recommend_price.py` | Benchmarks several network configurations, reports the best and (with `--save DIR`) persists it. See [Model recommender](#model-recommender-recommend_pricepy). |
| `baseline.py` | Non-neural references (mean predictor, linear regression, random forest) on the same features/split, so the network has something to beat. |
| `predict.py` | Loads a saved bundle and prices a single flat from CLI feature values. |
| `chat.py` | Conversational front-end: Claude asks you for the five features, then calls the saved model and tells you the price. See [Chat estimator](#chat-estimator-chatpy). |
| `export_web.py` | Exports a saved bundle to `../docs/model.json` for the browser demo (weights + scaler + price band, no TensorFlow.js runtime). |
| `dataset.py` / `metrics.py` | Shared dataset loading and regression scoring. |
| `notebook.ipynb` | Exploratory notebook mirroring `model.py`. |

## Features

`dataset.py` defines the honest feature set: `Habitaciones`, `Aseos`,
`Superficie`, `Parking`, `Colegios` → `Precio`. `Precio_m2` is **excluded by
default** because `precio ≈ precio_m2 × superficie` leaks the target;
`model.py` / `select_model.py` keep it (the thesis setup), and
`recommend_price.py` / `baseline.py` take `--keep-precio-m2` to opt back in.
Missing numeric values are filled with `1`.

## Model recommender (`recommend_price.py`)

1. **Load** – `dataset.load_xy` selects the feature columns by name (so a CSV with
   or without a leading index column both work).
2. **Preprocess** – features standardized with `StandardScaler` (fit on the
   training split only). The target stays on its raw € scale; predictions are
   clipped to a sane band (½·min … 1.5·max training price) so a diverging network
   cannot report absurd values.
3. **Benchmark** – trains 6 configurations from scratch on the same 70/30 split,
   each with `EarlyStopping` (patience 15, `restore_best_weights`):

   | config | hidden layers | optimizer | loss |
   | --- | --- | --- | --- |
   | `2x6 / adam / mse` | 6, 6 | adam | mse |
   | `4x6 / adam / mse` | 6, 6, 6, 6 | adam | mse |
   | `3x12 / adam / mse` | 12, 12, 12 | adam | mse |
   | `3x24 / adam / mae` | 24, 24, 24 | adam | mae |
   | `pyramid 32-16-8 / adam / huber` | 32, 16, 8 | adam | huber |
   | `3x24 / rmsprop / mse` | 24, 24, 24 | rmsprop | mse |

4. **Rank** – scores each on the held-out test set with **MAE, RMSE, MAPE and R²**
   and names the most recommended configuration.
5. **Recommend / save** – retrains the winner on the full dataset, prints the
   price for a sample flat (dataset medians) and, with `--save DIR`, writes
   `model.keras`, `scaler.joblib` and `metadata.json` for `predict.py`.

Tune from the CLI (`--epochs`, `--batch-size`, `--seed`, `--keep-precio-m2`,
`--save`) or by editing `CONFIGURATIONS`.

## Requirements

- Python 3.9+
- `pip install -r requirements.txt`

  `pandas` and `matplotlib` must be builds compatible with the installed `numpy`
  (a `numpy` 2.x / `pandas` 1.3 mix raises `numpy.dtype size changed`; an old
  `matplotlib` raises `_ARRAY_API not found`). `pip install -U pandas matplotlib`
  fixes both.

## Usage

```bash
cd keras_neural_network

python baseline.py                              # non-neural reference numbers
python model.py --epochs 150                    # train the thesis model
python select_model.py --optimizers SGD Adam    # cross-validate optimizers
python recommend_price.py --save model_dir      # benchmark, recommend, save
python predict.py model_dir --Habitaciones 3 --Aseos 2 --Superficie 90 \
    --Parking 0 --Colegios 9
```

On Windows, if `python` opens the Microsoft Store, use the launcher: `py model.py`.

Every script reads `finalDataset3.csv` from the current directory by default;
pass another file with `--dataset` (`model.py`, `select_model.py`) or as the
first positional argument (`recommend_price.py`, `baseline.py`).

## Chat estimator (`chat.py`)

A conversational front-end powered by Claude. It asks you for the five features
(rooms, bathrooms, m², parking, nearby schools), then calls the saved model via a
tool and reports the price — you never touch the CLI flags.

```bash
python recommend_price.py --save model_dir   # once: train and save the model
export ANTHROPIC_API_KEY=sk-ant-...          # or run `ant auth login`
python chat.py model_dir                      # --model claude-haiku-4-5 for a cheaper run
```

How it works: `chat.py` runs a short tool-use loop against the Anthropic
Messages API. Claude collects the inputs in natural language, calls the
`predict_price` tool (which runs `predict.py`'s model bundle locally), and turns
the result into a sentence. The API key and Claude usage are billed to you;
everything else stays on your machine.

