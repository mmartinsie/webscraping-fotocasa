# Browser demo

`index.html` is a single static page that estimates a Madrid flat's price. The
network is tiny, so `model.json` carries the raw weights + `StandardScaler` stats
+ the price-clip band, and `index.html` runs the forward pass in plain JavaScript
(no TensorFlow.js).

## Regenerate `model.json`

After retraining the model:

```bash
cd keras_neural_network
python recommend_price.py --save web_model
python export_web.py web_model -o ../docs/model.json
git add docs/model.json && git commit -m "Refresh demo model"
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
