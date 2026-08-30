# Reference data

## `precio_m2_distrito.csv`

Approximate **second-hand asking price per m²** for each of Madrid's 21 districts,
in euros. Ballpark **2024-2025** figures compiled from the public
Idealista / Fotocasa price indices, rounded to the nearest €100.

These are estimates, not an official series - update them from
<https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/> (venta,
Madrid capital, por distrito) or the Fotocasa Índice Inmobiliario.

Used by `webapp/pricing.py` (`estimate_by_district`) and, as `docs/districts.json`,
by the browser demo. It carries the **current** price level; the thesis network
(trained on ~2020 data, one-hot district) captures relative location differences
but from dated prices.

## `colegios_distrito.csv`

Number of nearby schools per district, taken from the thesis dataset (the value
is constant within each district there, so it's really a district attribute).
Both demos look this up from the district instead of asking the user for it, and
feed it to the neural network. `export_web.py` merges it into `districts.json` as
`{"District": {"eur_m2": ..., "colegios": ...}}`.
