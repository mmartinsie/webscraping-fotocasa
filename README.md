# Web Scraping Fotocasa

Este repositorio contiene todos los script desarrollado para la realización del trabajo final de máster "Categorización de los inmuebles de la ciudad de Madrid". Está compuesto por dos partes, por un lado está el rascado de información desde la página web de Fotocasa y, por otro lado, la red neuronal que precide el precio de un inmueble en función de la introducción de sus características. 

## Webscraping Fotocasa
Contenido en el directorio `/webscraping`, esta compuesto por tres script: 
- `main.py`
    - Este script contiene el programa principal que gestionará el rascado de la página web. Su objetivo principal es la aceptación de cookies, mediante la libreria *Selenium*; la iteractuación con las distintas páginas de la búsqueda y; por último, la llamada a la función `scrape_page()`.
- `home.py`
    - Este script contiene la definición de la clase que representa un inmueble. Esta clase está compuesta por los atributos: `price`, `district`, `rooms`, `baths`, `size`, `floor`, `url`, `type` y `parking`.
- `page_url.py`.
    - Este script realiza la funcionalidad de rascado para cada una de las páginas proporcionadas desde el script principal. 

## Red neuronal
Existen dos desarrollos de una red neuronal en este repositorio. El primero de ellos, en el directorio `/neural_network`, se encuentra obsoleto y sin uso, pues fue una primera iteración. En segundo lugar está el directorio `/keras_neural_network`, el cual contiene el desarrollo final de la red neuronal empleando la biblioteca de Python, *keras*.

## Autores ✒️
Este repositorio está realizado por:
- Esther Gabay Diaz
- Adrián Caminmo Muñoz - [adrian98cm](https://github.com/adrian98cm)
- María González de la Llana Domarco
- Manuel Martín Sierra - [mmartinsie](https://github.com/mmmartinsie)
- Miriam Ramón González - [MiriamRG13](https://github.com/MiriamRG13)

## Referencias
A continuación se muestran las web que se ha empleado para el desarrollo del proyecto.
- [Ejemplo de la página web Edureka](https://www.edureka.co/blog/web-scraping-with-python/)
- [Ander Fernández - Cómo programar una red neuronal desde 0 en Python](https://anderfernandez.com/blog/como-programar-una-red-neuronal-desde-0-en-python/)
- [W3Schools - Python](https://www.w3schools.com/python/python_classes.asp)
