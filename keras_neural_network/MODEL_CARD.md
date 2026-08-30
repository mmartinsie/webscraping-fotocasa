# Model card — flat price predictor

## Overview

A small feed-forward network that estimates the sale price of a flat in Madrid
from five numeric features. Produced by
[`recommend_price.py`](recommend_price.py), which cross-validates six
configurations and retrains the winner on the full dataset.

- **Type:** regression (single linear output, EUR).
- **Winning architecture:** `Input(5) → Dense(24, relu) × 3 → Dense(1, linear)`,
  Adam, MAE loss, early stopping.
- **Preprocessing:** `StandardScaler` on the features (fit on training data);
  missing values filled with the column median; predictions clipped to
  `[0.5·min, 1.5·max]` of the training prices.
- **Artifact:** `model.keras` + `scaler.joblib` + `metadata.json`
  (features, medians, price band, CV metrics). `export_web.py` also exports the
  weights to `docs/model.json` for an in-browser NumPy forward pass.

## Intended use

Rough, educational price *orientation* and district comparison. **Not** a
valuation, appraisal, or financial advice. The demos show this number only as a
secondary reference next to a `district €/m² × m²` estimate.

## Training data

`finalDataset3.csv` — ~8,300 Madrid flats scraped from Fotocasa **circa 2020**,
cleaned and joined with a per-district school count. Features:
`Habitaciones, Aseos, Superficie, Parking, Colegios`. Target: `Precio`.
`Precio_m2` and `Distrito`/`Tipo` are dropped (the first leaks the target, the
categoricals are not encoded here). See [`DATA.md`](DATA.md).

## Evaluation

3-fold cross-validation, mean over folds (varies slightly by seed):

| metric | value |
| --- | ---: |
| MAE | ~193,000 € |
| RMSE | ~410,000 € |
| MAPE | ~40 % |
| R² | ~0.64 |

Reference points on the same split:

| model | MAE | R² |
| --- | ---: | ---: |
| mean predictor | ~421k € | 0.00 |
| linear regression | ~204k € | 0.63 |
| random forest | ~170k € | 0.68 |
| random forest + one-hot district | ~170k € | 0.70 |

## Limitations and risks

- **Dated.** Training prices are ~2020; Madrid has risen sharply since, so the
  *level* is low. Use it for relative comparison, not absolute value.
- **No location.** `Distrito` is dropped, so the network cannot tell Salamanca
  from Villaverde. The demos compensate with the €/m² table.
- **Outperformed.** A plain random forest beats it on this data — the network is
  kept as a thesis artifact, not because it is the best model.
- **Coverage.** Trained on listings that were live in 2020; long tails (very
  large or very cheap properties) are thin and clipped.
- **Asking prices**, not closing prices, and only flats/houses on one portal.

## Ethical considerations

Automated price estimates can influence negotiations and reinforce existing
price gaps between districts. This model is a teaching artifact; it should not
be used to set prices or assess individuals' properties.
