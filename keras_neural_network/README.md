# Keras neural network

Final implementation of the model that predicts the sale price of a Madrid
property from its characteristics, built with [Keras](https://keras.io/).

## Contents

| File | Purpose |
| --- | --- |
| `model.py` | Trains the final model: a `Sequential` network with four hidden `Dense(6, relu)` layers and a `Dense(1, relu)` output, SGD optimizer, `mean_squared_logarithmic_error` loss, 150 epochs with a 30% validation split. Prints the minimum MSLE and contains commented-out code to plot history and to predict interactively. |
| `select_model.py` | Hyper-parameter search with `GridSearchCV` (`KerasClassifier` wrapper) over the optimizer (`SGD` vs `RMSprop`). |
| `recommend_price.py` | Benchmarks several network configurations and reports the one that predicts price best. See [Model recommender](#model-recommender-recommend_pricepy) below. |
| `Notebook .ipynb` | Exploratory notebook with the same modelling workflow. |
| `finalDataset3.csv` / `pisos.csv` | Cleaned dataset used for training. Columns: `Precio`, `Precio_m2`, `Habitaciones`, `Aseos`, `Superficie`, `Parking`, `Colegios`, `Tipo`, `Distrito`. |
| `finalDataset.csv` | Earlier, wider version of the dataset. |
| `buildings_information.csv` | Raw output of the scraper (`Precio`, `Distrito`, `Tipo`, `Habitaciones`, `Aseos`, `Superficie`, `Planta`, `Parking`, `URL`). |

## Features used by the model

`model.py` and `select_model.py` drop `Tipo` and `Distrito`, fill missing values
with `1`, and use 6 numeric predictors (`Precio_m2`, `Habitaciones`, `Aseos`,
`Superficie`, `Parking`, `Colegios`) to predict `Precio`. `recommend_price.py`
uses the same 6 predictors but additionally standardizes them.

## Model recommender (`recommend_price.py`)

`recommend_price.py` automates picking a network for the price-prediction task:

1. **Load & clean** – reads the dataset, fills missing values with `1`, keeps the
   6 numeric predictors and `Precio` as target.
2. **Preprocess** – features are standardized with `StandardScaler` (fit on the
   training split only). The target is kept on its raw € scale; predictions are
   clipped to a sane price band (½·min … 1.5·max training price) so a diverging
   network cannot report absurd values.
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

4. **Rank** – scores each on the held-out test set with **MAE, RMSE, MAPE and
   R²**, prints the ranking (best MAE first) and names the most recommended
   configuration.
5. **Recommend a price** – retrains the winner on the full dataset and prints the
   recommended price for a sample flat (the column-wise medians of the dataset).
   Call `recommend_price({...})` with your own feature values to price a specific
   flat.

Edit the module-level constants to tune it: `CONFIGURATIONS` (candidate
networks), `FEATURES` (remove `Precio_m2` for a harder, more realistic
benchmark), `EPOCHS`, `BATCH_SIZE`, `RANDOM_SEED`.

## Requirements

- Python 3.9 (tested; 3.8 also works)
- Python packages:

  ```bash
  pip install tensorflow keras scikit-learn numpy "pandas>=2.2" matplotlib
  ```

  `pandas` must be a build compatible with the installed `numpy` (a `numpy` 2.x /
  `pandas` 1.3 mix raises `numpy.dtype size changed`).

## Usage

```bash
cd keras_neural_network

# train the final model
python model.py

# run the optimizer grid search
python select_model.py

# benchmark configs and get a recommended price
python recommend_price.py [path/to/dataset.csv]
```

On Windows, if `python` opens the Microsoft Store, use the launcher: `py model.py`.

`model.py` and `select_model.py` read `finalDataset3.csv` from the current
directory (change `csv_route` at the top of each file to use a different
dataset). `recommend_price.py` defaults to `finalDataset3.csv` and accepts an
optional dataset path as its first argument.
