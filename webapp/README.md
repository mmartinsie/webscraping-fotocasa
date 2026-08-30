# Conversational web app

`streamlit_app.py` is a chat where **Google Gemini** collects the flat's features
and calls a tool that runs the model from [`../docs/model.json`](../docs/model.json)
(NumPy forward pass in `pricing.py` - no TensorFlow, fast cold start).

Free to run: Gemini's free tier covers a demo's traffic, and visitors don't need
a key of their own.

## Deploy on Streamlit Community Cloud

1. Get a free API key at <https://aistudio.google.com/apikey>.
2. On <https://share.streamlit.io>, "New app" → this repo, **main file**
   `webapp/streamlit_app.py`.
3. In the app's **Settings → Secrets**, add:

   ```toml
   GEMINI_API_KEY = "AIza..."
   ```

4. Deploy. It redeploys automatically on every push.

## Run locally

```bash
pip install -r webapp/requirements.txt
GEMINI_API_KEY=AIza... streamlit run webapp/streamlit_app.py
```

## Notes

- The tool's parameters (`habitaciones`, `aseos`, `superficie`, `parking`,
  `colegios`) are fixed to the current `docs/model.json` feature set. If you
  retrain with different features, update `estimar_precio` and `SYSTEM`.
- Model defaults to `gemini-flash-latest` (an alias Google keeps current, so it
  survives version retirements). Pick another in the sidebar, or pin one with a
  `GEMINI_MODEL` secret / env var. On a 404 the app lists the models your key can
  use.
- `pricing.predict_price` is covered by `tests/test_pricing.py`; the Streamlit +
  Gemini glue is not unit-tested (needs the runtime and a live key).
