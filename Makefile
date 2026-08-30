# Convenience targets. Use `py` on Windows if `python` opens the Store stub.
PYTHON ?= python

.PHONY: lint test check demo-model dataset

lint:
	ruff check .
	ruff format --check .

test:
	pytest -q

check: lint test
	$(PYTHON) -m compileall -q webscraping neural_network keras_neural_network prepare_dataset.py

# Rebuild the browser demo's model (docs/model.json) from the current code.
demo-model:
	cd keras_neural_network && $(PYTHON) recommend_price.py --save web_model
	cd keras_neural_network && $(PYTHON) export_web.py web_model -o ../docs/model.json

# Clean the raw scraper CSV into a model-ready dataset.
dataset:
	$(PYTHON) prepare_dataset.py webscraping/buildings_information.csv -o dataset.csv
