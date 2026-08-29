# Keras neural network

Final implementation of the model that predicts the sale price of a Madrid
property from its characteristics, built with [Keras](https://keras.io/).

## Contents

| File | Purpose |
| --- | --- |
| `model.py` | Trains the final model: a `Sequential` network with four hidden `Dense(6, relu)` layers and a `Dense(1, relu)` output, SGD optimizer, `mean_squared_logarithmic_error` loss, 150 epochs with a 30% validation split. Prints the minimum MSLE and contains commented-out code to plot history and to predict interactively. |
| `select_model.py` | Hyper-parameter search with `GridSearchCV` (`KerasClassifier` wrapper) over the optimizer (`SGD` vs `RMSprop`). |
| `recommend_price.py` | Trains and compares several network configurations (layers, optimizer, loss) on a held-out test set, ranks them by MAE / RMSE / MAPE / R², prints the most recommended one, then retrains the winner on the full dataset and outputs a recommended price for a sample flat. Run `python recommend_price.py [dataset.csv]`. |
| `Notebook .ipynb` | Exploratory notebook with the same modelling workflow. |
| `finalDataset3.csv` / `pisos.csv` | Cleaned dataset used for training. Columns: `Precio`, `Precio_m2`, `Habitaciones`, `Aseos`, `Superficie`, `Parking`, `Colegios`, `Tipo`, `Distrito`. |
| `finalDataset.csv` | Earlier, wider version of the dataset. |
| `buildings_information.csv` | Raw output of the scraper (`Precio`, `Distrito`, `Tipo`, `Habitaciones`, `Aseos`, `Superficie`, `Planta`, `Parking`, `URL`). |

## Features used by the model

`model.py` and `select_model.py` drop `Tipo` and `Distrito`, fill missing values
with `1`, and use 6 numeric predictors (`Precio_m2`, `Habitaciones`, `Aseos`,
`Superficie`, `Parking`, `Colegios`) to predict `Precio`.

## Requirements

- Python 3.8
- Python packages:

  ```bash
  pip install tensorflow keras scikit-learn numpy pandas matplotlib
  ```

## Usage

```bash
cd keras_neural_network

# train the final model
python model.py

# run the optimizer grid search
python select_model.py
```

Both scripts read `finalDataset3.csv` from the current directory (change
`csv_route` at the top of each file to use a different dataset).
