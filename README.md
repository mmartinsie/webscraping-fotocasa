# Web Scraping Fotocasa

This repository contains all the scripts developed for the Master's thesis
*"Categorization of real-estate properties in the city of Madrid"*. It is made up
of two parts: on one side, the scraping of information from the Fotocasa website
and, on the other side, the neural network that predicts the price of a property
based on its characteristics.

## Web scraping Fotocasa

Contained in the `/webscraping` directory, it is made up of three scripts:

- `main.py`
    - Command-line entry point. Drives Firefox through *Selenium* (accept
      cookies, page through the search results, read each card's district),
      calls `scrape_listing()` per listing and writes the CSV.
- `listing.py`
    - `scrape_listing()` downloads and parses a single listing page and returns
      a populated `Home`.
- `home.py`
    - `Home` dataclass representing one property (`url`, `district`, `price`,
      `property_type`, `rooms`, `baths`, `size`, `floor`, `parking`).

See [`webscraping/README.md`](webscraping/README.md) for details.

## Neural network

There are two neural-network implementations in this repository. The first one,
in the `/neural_network` directory, is obsolete and unused — it was a first
iteration written from scratch with NumPy. The second one, in the
`/keras_neural_network` directory, is the final implementation using the Python
library *Keras*. That directory also includes `recommend_price.py`, which
benchmarks several network configurations, reports the one that predicts price
best and prints a recommended price for a flat.

See [`neural_network/README.md`](neural_network/README.md) and
[`keras_neural_network/README.md`](keras_neural_network/README.md) for details.

## Authors ✒️

This repository was created by:

- Esther Gabay Diaz
- Adrián Camino Muñoz - [adrian98cm](https://github.com/adrian98cm)
- María González de la Llana Domarco
- Manuel Martín Sierra - [mmartinsie](https://github.com/mmartinsie)
- Miriam Ramón González - [MiriamRG13](https://github.com/MiriamRG13)

## References

The websites used during the development of the project:

- [Web scraping with Python example — Edureka](https://www.edureka.co/blog/web-scraping-with-python/)
- [Ander Fernández — How to program a neural network from scratch in Python](https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/)
- [W3Schools — Python classes](https://www.w3schools.com/python/python_classes.asp)
- [Machine Learning Mastery — Your first neural network in Python with Keras](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)
