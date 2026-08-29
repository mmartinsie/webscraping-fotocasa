# Web Scraping Fotocasa

This repository contains all the scripts developed for the Master's thesis
*"Categorization of real-estate properties in the city of Madrid"*. It is made up
of two parts: on one side, the scraping of information from the Fotocasa website
and, on the other side, the neural network that predicts the price of a property
based on its characteristics.

## Web scraping Fotocasa

Contained in the `/webscraping` directory, it is made up of three scripts:

- `main.py`
    - Main program that drives the scraping of the website. Its main goals are
      accepting cookies through the *Selenium* library, navigating the different
      pages of the search results and, finally, calling the `scrap_page()`
      function.
- `home.py`
    - Definition of the class that represents a property. The class has the
      attributes: `price`, `district`, `rooms`, `baths`, `size`, `floor`, `url`,
      `type` and `parking`.
- `page_url.py`
    - Runs the scraping logic for each of the pages provided by the main script.

See [`webscraping/README.md`](webscraping/README.md) for details.

## Neural network

There are two neural-network implementations in this repository. The first one,
in the `/neural_network` directory, is obsolete and unused — it was a first
iteration written from scratch with NumPy. The second one, in the
`/keras_neural_network` directory, is the final implementation using the Python
library *Keras*.

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
