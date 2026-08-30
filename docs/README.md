# Browser demo

`index.html` is a single static page (English / Spanish toggle) that estimates a
Madrid flat's price. It leads with `district €/m² × m²` from `districts.json`
(`{eur_m2, colegios}` per district, ~2024 figures) and shows the thesis network's
number as a reference; the school count fed to the network is taken from the
district, not entered by the user. The network is tiny, so `model.json` carries
the raw weights + `StandardScaler` stats + the price-clip band, and the page runs
the forward pass in plain JavaScript (no TensorFlow.js).

## Regenerate `model.json` / `districts.json`

`export_web.py` writes both (weights from the bundle, €/m² table from
`data/precio_m2_distrito.csv`):

```bash
cd keras_neural_network
python recommend_price.py --save web_model
python export_web.py web_model -o ../docs/model.json
git add docs/model.json docs/districts.json && git commit -m "Refresh demo data"
```

## Local preview

`fetch` needs a server (not `file://`):

```bash
python -m http.server -d docs 8000   # then open http://localhost:8000
```

## Deployment

GitHub Pages serves this folder directly - no workflow needed. Enable it once in
**Settings → Pages → Build and deployment → Source: "Deploy from a branch" →
Branch: `master` `/docs`**. Every push to `docs/` then republishes it. Live at
`https://mmartinsie.github.io/webscraping-fotocasa/`.
