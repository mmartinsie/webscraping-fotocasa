# Web scraping Fotocasa

Scrapes the listings of properties for sale in Madrid from
[fotocasa.es](https://www.fotocasa.es/) and stores the extracted data in a CSV
file (`buildings_information.csv`).

## Scripts

| File | Purpose |
| --- | --- |
| `main.py` | Entry point. Opens Firefox with Selenium, loads each results page, accepts the cookie banner, scrolls to force lazy-loaded cards to render, reads the district of every card and calls `scrap_page()` for each listing URL. |
| `page_url.py` | `scrap_page(page_url, district)` — downloads a single listing page with `requests`, parses it with BeautifulSoup, fills a `Home` object (price, rooms, baths, size, floor, type, parking) and appends a row to `buildings_information.csv`. |
| `home.py` | `Home` class representing one property, with the attributes `price`, `district`, `rooms`, `baths`, `size`, `floor`, `url`, `type`, `parking` and a `toString()` helper that prints them. |

## Requirements

- Python 3.8
- [Firefox](https://www.mozilla.org/firefox/) and
  [geckodriver](https://github.com/mozilla/geckodriver/releases)
- Python packages:

  ```bash
  pip install selenium beautifulsoup4 requests
  ```

## Usage

1. Install geckodriver and update the path in `main.py`:

   ```python
   driver = webdriver.Firefox(executable_path='C:/WebDriver/bin/geckodriver.exe')
   ```

2. Adjust the page range of the loop in `main.py` (`for k in range(1, 2)`) to
   scrape more result pages.
3. Run:

   ```bash
   cd webscraping
   python main.py
   ```

The output CSV `buildings_information.csv` is written to the current working
directory with the columns: `Precio`, `Distrito`, `Tipo de inmueble`,
`Habitaciones`, `Aseos`, `Superficie`, `Planta`, `Parking`, `URL`.

## Notes

- The scraper depends on Fotocasa's HTML structure and XPaths as of the time it
  was written; the site's markup changes often, so selectors may need updating.
- Be respectful with request rate and check Fotocasa's terms of service before
  running large scrapes.
