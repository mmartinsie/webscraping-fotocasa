# Reference data

## `precio_m2_distrito.csv`

Approximate **second-hand asking price per m²** for each of Madrid's 21 districts,
in euros. Ballpark **2024-2025** figures compiled from the public
Idealista / Fotocasa price indices, rounded to the nearest €100.

These are estimates, not an official series - update them from
<https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/> (venta,
Madrid capital, por distrito) or the Fotocasa Índice Inmobiliario.

Used by `webapp/pricing.py` (`estimate_by_district`) and, as `docs/districts.json`,
by the browser demo. It gives the location + "current price level" signal that
the thesis neural network (trained on ~2020 data, district dropped) does not
have.
