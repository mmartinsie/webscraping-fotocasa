# Web scraping Fotocasa

Scrapes the listings of properties for sale in Madrid from
[fotocasa.es](https://www.fotocasa.es/) and stores the extracted data in a CSV
file (`buildings_information.csv`).

## Scripts

| File | Purpose |
| --- | --- |
| `main.py` | Entry point / CLI. Opens Firefox with Selenium, walks the results pages (accept cookies, scroll for lazy-loaded cards, read each card's district), calls `scrape_listing()` per listing and writes the CSV. |
| `listing.py` | `scrape_listing(session, url, district) -> Home \| None` — downloads one listing page with a shared `requests.Session`, parses it with BeautifulSoup and returns a populated `Home` (no side effects). |
| `home.py` | `Home` dataclass (`url`, `district`, `price`, `property_type`, `rooms`, `baths`, `size`, `floor`, `parking`) plus `to_csv_row()` and the shared `CSV_HEADERS`. |

## Requirements

- Python 3.9+
- [Firefox](https://www.mozilla.org/firefox/) and
  [geckodriver](https://github.com/mozilla/geckodriver/releases) on `PATH`
- Python packages (from the repo root):

  ```bash
  pip install -e .                       # makes `import webscraping...` work
  pip install -r webscraping/requirements.txt
  ```

## Usage

```bash
python webscraping/main.py --pages 5 --output buildings_information.csv
```

Point the scraper at geckodriver with `--geckodriver PATH` or the
`GECKODRIVER_PATH` environment variable (otherwise it is looked up on `PATH`).

| Option | Default | Meaning |
| --- | --- | --- |
| `--pages` | `1` | number of search-results pages to scrape |
| `--start-page` | `1` | first results page (1-based) |
| `--geckodriver` | `$GECKODRIVER_PATH` | path to the geckodriver binary |
| `--output` | `buildings_information.csv` | output CSV path |
| `--resume` | off | append to `--output`, skipping listing URLs it already contains |
| `--headless` | off | run Firefox without a window |
| `--delay` | `5.0` | seconds between listing requests |
| `--log-level` | `INFO` | logging verbosity |

Rows are flushed to the CSV as they are scraped, so a crashed run can be picked
up again with `--resume`.

The output CSV is written with the columns `Precio`, `Distrito`, `Tipo`,
`Habitaciones`, `Aseos`, `Superficie`, `Planta`, `Parking`, `URL` (Spanish
names, matching the Fotocasa fields and the existing `buildings_information.csv`).

## Notes

- The CSS classes and XPaths are centralised as constants at the top of `main.py`
  and `listing.py`. They reflect Fotocasa's DOM at the time of the thesis and are
  the first thing to check if the scraper stops finding data.
- Be respectful with the request rate and review Fotocasa's terms of service
  before running large scrapes.
