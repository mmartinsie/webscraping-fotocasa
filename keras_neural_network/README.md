# Keras neural network

Final implementation of the model that predicts the sale price of a Madrid
property from its characteristics, built with [Keras](https://keras.io/).

See [`DATA.md`](DATA.md) for the datasets and their columns, and
[`MODEL_CARD.md`](MODEL_CARD.md) for the trained model's intended use, metrics
and limitations.

## Contents

| File | Purpose |
| --- | --- |
| `model.py` | Trains the thesis model (CLI): a `Sequential` network with four hidden `Dense(6, relu)` layers and a `Dense(1, relu)` output, SGD + `mean_squared_logarithmic_error`, 150 epochs, 30% validation split. Prints the minimum MSLE; `--plot` charts the history. |
| `select_model.py` | Compares optimizers (`SGD`, `RMSprop`, `Adam` by default) with k-fold cross-validation, scoring by mean squared error, and reports the best. |
| `recommend_price.py` | Cross-validates several network configurations, reports the best and (with `--save DIR`) persists it; `--with-district` adds the one-hot `Distrito`. See [Model recommender](#model-recommender-recommend_pricepy). |
| `baseline.py` | Non-neural references (mean predictor, linear regression, random forest) on the same features/split. `--with-district` adds one-hot `Distrito`. |
| `predict.py` | Loads a saved bundle and prices a single flat (`--set NAME=VALUE` / `--json`, e.g. `--set Distrito=Retiro`); missing numeric features default to the stored median. |
| `chat.py` | Conversational front-end: Claude asks you for the model's features (incl. district) and calls the saved model. See [Chat estimator](#chat-estimator-chatpy). |
| `export_web.py` | Exports a saved bundle to `../docs/model.json` / `districts.json` for the browser demo (weights + scaler + district categories, no TensorFlow.js runtime). |
| `dataset.py` / `metrics.py` | Shared dataset loading (`load_xy`, `one_hot_district`) and regression scoring. |
| `notebook.ipynb` | Exploratory notebook mirroring `model.py`. |

## Features

`dataset.py` defines the honest feature set: `Habitaciones`, `Aseos`,
`Superficie`, `Parking`, `Colegios` → `Precio`. `Precio_m2` is **excluded by
default** because `precio ≈ precio_m2 × superficie` leaks the target;
`model.py` / `select_model.py` keep it (the thesis setup), and
`recommend_price.py` / `baseline.py` take `--keep-precio-m2` to opt back in.
Missing numeric values are filled with the **column median** (`dataset.load_xy`);
the medians are saved in the bundle so `predict.py` / `chat.py` use them as
defaults for anything you don't provide.

`recommend_price.py --with-district` one-hot encodes `Distrito` (21 columns) and
appends it, so the network can actually learn location. The saved bundle records
`district_categories` (the column order) and `predict.py` / `webapp` / the browser
rebuild the one-hot from the chosen district. The deployed demo model is trained
this way.

## Model recommender (`recommend_price.py`)

1. **Load** – `dataset.load_xy` selects the numeric columns by name and
   median-fills gaps; `--with-district` appends `one_hot_district` (21 columns).
2. **Cross-validate** – each of the 6 configurations below is trained from scratch
   on every fold of a `--folds`-way `KFold` (default 3). A fresh scaler per fold
   standardizes the **numeric** columns only (the one-hot is passed through);
   `EarlyStopping` (patience 15) runs on an inner validation split. Configs are
   ranked by **mean CV MAE** (RMSE / MAPE / R² reported too).

   | config | hidden layers | optimizer | loss |
   | --- | --- | --- | --- |
   | `2x6 / adam / mse` | 6, 6 | adam | mse |
   | `4x6 / adam / mse` | 6, 6, 6, 6 | adam | mse |
   | `3x12 / adam / mse` | 12, 12, 12 | adam | mse |
   | `3x24 / adam / mae` | 24, 24, 24 | adam | mae |
   | `pyramid 32-16-8 / adam / huber` | 32, 16, 8 | adam | huber |
   | `3x24 / rmsprop / mse` | 24, 24, 24 | rmsprop | mse |

3. **Recommend / save** – retrains the winner on the full dataset (same inner
   validation split + early stopping), prints the price for a sample flat and,
   with `--save DIR`, writes `model.keras`, `scaler.joblib` and `metadata.json`
   (`numeric_features`, `district_categories`, medians, price band, CV metrics)
   for `predict.py`.

Tune from the CLI (`--folds`, `--epochs`, `--batch-size`, `--seed`,
`--with-district`, `--keep-precio-m2`, `--save`) or by editing `CONFIGURATIONS`.

## Requirements

- Python 3.9+
- From the repo root: `pip install -e .` (makes `import keras_neural_network...`
  work) then `pip install -r keras_neural_network/requirements.txt`

  `pandas` and `matplotlib` must be builds compatible with the installed `numpy`
  (a `numpy` 2.x / `pandas` 1.3 mix raises `numpy.dtype size changed`; an old
  `matplotlib` raises `_ARRAY_API not found`). `pip install -U pandas matplotlib`
  fixes both.

## Usage

```bash
cd keras_neural_network   # so the scripts find finalDataset3.csv

python baseline.py --with-district             # non-neural reference numbers
python model.py --epochs 150                   # train the thesis model
python select_model.py --optimizers SGD Adam   # cross-validate optimizers
python recommend_price.py --with-district --save model_dir   # CV, recommend, save
python predict.py model_dir --set Superficie=90 --set Distrito=Retiro
```

On Windows, if `python` opens the Microsoft Store, use the launcher: `py model.py`.

Imports are package-qualified (`from keras_neural_network.dataset import …`), so
`pip install -e .` is what makes the scripts runnable — the `cd` is only so they
find `finalDataset3.csv` (or pass `--dataset`).
pass another file with `--dataset` (`model.py`, `select_model.py`) or as the
first positional argument (`recommend_price.py`, `baseline.py`).

## Chat estimator (`chat.py`)

A conversational front-end powered by Claude. It reads the feature list and
medians straight from the saved bundle, asks you for each one, then calls the
model via a tool and reports the price — you never touch the CLI flags, and a
model retrained on different features just works.

```bash
python recommend_price.py --with-district --save model_dir   # once: train + save
export ANTHROPIC_API_KEY=sk-ant-...          # or run `ant auth login`
python chat.py model_dir                      # --model claude-haiku-4-5 for a cheaper run
```

How it works: `chat.py` runs a short tool-use loop against the Anthropic
Messages API. Claude collects the inputs in natural language, calls the
`predict_price` tool (which runs `predict.py`'s model bundle locally), and turns
the result into a sentence. The API key and Claude usage are billed to you;
everything else stays on your machine.

