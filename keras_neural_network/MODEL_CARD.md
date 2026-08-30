# Model card — flat price predictor

## Overview

A small feed-forward network that estimates the sale price of a flat in Madrid
from five numeric features plus a one-hot of the district. Produced by
[`recommend_price.py --with-district`](recommend_price.py), which cross-validates
six configurations and retrains the winner on the full dataset.

- **Type:** regression (single linear output, EUR).
- **Winning architecture:** `Input(26) → Dense(24, relu) × 3 → Dense(1, linear)`,
  RMSprop, MSE loss, early stopping.
- **Inputs:** `Habitaciones, Aseos, Superficie, Parking, Colegios` +
  one-hot(`Distrito`, 21 categories).
- **Preprocessing:** `StandardScaler` on the *numeric* columns only (the one-hot
  is passed through); missing numeric values filled with the column median;
  predictions clipped to `[0.5·min, 1.5·max]` of the training prices.
- **Artifact:** `model.keras` + `scaler.joblib` + `metadata.json`
  (`numeric_features`, `district_categories`, medians, price band, CV metrics).
  `export_web.py` exports the weights to `docs/model.json` for an in-browser
  NumPy forward pass.

## Intended use

Rough, educational price *orientation* and district comparison. **Not** a
valuation, appraisal, or financial advice. The demos show this number only as a
secondary reference next to a `district €/m² × m²` estimate.

## Training data

`finalDataset3.csv` — ~8,300 Madrid flats scraped from Fotocasa **circa 2020**,
cleaned and joined with a per-district school count. `Precio_m2` and `Tipo` are
dropped (the first leaks the target). See [`DATA.md`](DATA.md).

## Evaluation

3-fold cross-validation, mean over folds (varies slightly by seed):

| metric | value |
| --- | ---: |
| MAE | ~200,000 € |
| RMSE | ~396,000 € |
| MAPE | ~46 % |
| R² | ~0.66 |

Reference points on the same task:

| model | MAE | R² |
| --- | ---: | ---: |
| mean predictor | ~421k € | 0.00 |
| linear regression | ~204k € | 0.63 |
| random forest | ~170k € | 0.68 |
| random forest + one-hot district | ~170k € | 0.70 |
| **this network (5 numeric + district)** | **~200k €** | **~0.66** |
| same network without district | ~193k € | 0.64 |

Adding the district one-hot makes the network *location-aware* (its estimate now
moves with the zone) without hurting the aggregate metrics, but a plain random
forest still wins.

## Limitations and risks

- **Dated.** Training prices are ~2020; Madrid has risen sharply since. Use it
  for relative comparison, not absolute value — the demos lead with a ~2024
  €/m² estimate for that reason.
- **Muted location effect.** With unscaled 0/1 one-hot columns and a small net,
  the learned per-district premium is real but weaker than the €/m² table's.
- **Outperformed.** A random forest beats it on this data — the network is kept
  as a thesis artifact, not because it is the best model.
- **Asking prices**, not closing prices; flats/houses on one portal; long tails
  (very large or very cheap properties) are thin and clipped.

## Ethical considerations

Automated price estimates can influence negotiations and reinforce price gaps
between districts. This model is a teaching artifact; it should not be used to
set prices or assess individuals' properties.
