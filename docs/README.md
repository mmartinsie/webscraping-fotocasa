# Browser demo

`index.html` is a single static page (English / Spanish toggle) that estimates a
Madrid flat's price. It leads with `district €/m² × m²` from `districts.json`
(~2024 figures) and shows the thesis network's number as a reference. The network
is tiny, so `model.json` carries the raw weights + `StandardScaler` stats + the
price-clip band, and the page runs the forward pass in plain JavaScript (no
TensorFlow.js).

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

`.github/workflows/pages.yml` publishes this folder to GitHub Pages on every push
that touches `docs/`. Enable it once in **Settings → Pages → Build and deployment
→ Source → GitHub Actions**. Live at
`https://mmartinsie.github.io/webscraping-fotocasa/`.
