# Datasets

## Provenance

```
fotocasa.es ──(webscraping/main.py)──▶ buildings_information.csv
                                              │
                              (prepare_dataset.py + external "schools" join)
                                              ▼
                                        finalDataset3.csv  ◀── used by the model scripts
```

The `schools` join (`Colegios` column) was done outside this repo in the original
thesis pipeline (an R script, hence the `fullDataset.*` column names still
visible in `finalDataset.csv`). `prepare_dataset.py` reproduces every step except
that join; pass `--schools schools.csv` if you have the school counts per
district.

## Files

| File | Rows | Notes |
| --- | ---: | --- |
| `buildings_information.csv` | ~14k | Raw scraper output. Schema below. Empty cells where Fotocasa did not show a value. |
| `finalDataset3.csv` | ~8.3k | Cleaned + schools-joined. **This is the dataset the model scripts default to.** Has an unnamed leading index column. |
| `pisos.csv` | ~7k | Same schema as `finalDataset3.csv`, a smaller/earlier cut. |
| `finalDataset.csv` | ~12k | Earlier, wider version with extra one-hot columns (`fullDataset.*` names). Not used by any script; kept for reference. |

## Schema — `buildings_information.csv`

| Column | Type | Meaning |
| --- | --- | --- |
| `Precio` | int (EUR) | Asking price. |
| `Distrito` | str | Madrid district (e.g. `Salamanca`, `Ciudad Lineal`). |
| `Tipo` | str | Property type (`Piso`, `Ático`, `Casa adosada`, ...). |
| `Habitaciones` | int | Number of rooms. |
| `Aseos` | int | Number of bathrooms. |
| `Superficie` | int (m²) | Floor area. |
| `Planta` | int | Floor number (blank for houses / ground floor). |
| `Parking` | `1` / blank | `1` if the listing includes parking. |
| `URL` | str | Listing URL (used as the dedup / `--resume` key). |

## Schema — `finalDataset3.csv` / `pisos.csv`

Leading unnamed column = pandas row index. Then:

| Column | Type | Meaning |
| --- | --- | --- |
| `Precio` | int (EUR) | Target variable. |
| `Precio_m2` | int (EUR/m²) | `Precio / Superficie`. **Leaks the target** — excluded by `recommend_price.py` / `baseline.py` by default, kept by `model.py`. |
| `Habitaciones`, `Aseos`, `Superficie`, `Parking` | int | As above. |
| `Colegios` | int | Number of schools near the property (external data). |
| `Tipo` | str | Dropped by every model script before training. |
| `Distrito` | str | Dropped by `model.py` / `select_model.py`; `recommend_price.py --with-district` one-hot encodes it. |

Model scripts (`dataset.load_xy`) fill any missing numeric value with that
column's median.
