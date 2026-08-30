"""Scrape Madrid property listings from Fotocasa into a CSV file.

Drives Firefox with Selenium to walk the search-results pages, then downloads and
parses each listing with :mod:`listing`. Rows are written to the output CSV as
they are scraped, so an interrupted run can be continued with ``--resume``.

    python main.py --pages 5 --output buildings_information.csv
    python main.py --pages 5 --output buildings_information.csv --resume

A geckodriver binary is required (https://github.com/mozilla/geckodriver); pass
its path with ``--geckodriver`` or the ``GECKODRIVER_PATH`` environment variable.

The XPaths/classes below reflect Fotocasa's DOM at the time the thesis was
written and will need revisiting if the site layout has changed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import os
import time
from collections.abc import Iterator

import requests
from requests.adapters import HTTPAdapter
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from urllib3.util.retry import Retry

from home import CSV_HEADERS, Home
from listing import scrape_listing

logger = logging.getLogger("fotocasa.scraper")

SEARCH_URL_TEMPLATE = (
    "https://www.fotocasa.es/es/comprar/viviendas/madrid-capital/todas-las-zonas/l/{page}"
    "?combinedLocationIds=724%2C14%2C28%2C173%2C0%2C28079%2C0%2C0%2C0"
    "&latitude=40.4096&longitude=-3.6862"
)
COOKIE_ACCEPT_XPATH = "/html/body/div[3]/div/div/footer/div/button[2]"
CARD_LINK_CLASS = "re-Card-link"
# District label read *relative to each card* (first <h3> inside it), so the link
# and its district always come from the same node.
DISTRICT_RELATIVE_XPATH = ".//h3"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
SCROLL_PAUSE_TIME = 0.5
PAGE_LOAD_PAUSE = 5
MAX_SCROLLS = 40

# Madrid districts whose names are more than one word; the card label ends with
# the last N words instead of just the last one.
MULTIWORD_DISTRICTS = {
    "Vallecas": 3,  # "Puente de Vallecas" / "Villa de Vallecas"
    "Lineal": 2,  # "Ciudad Lineal"
    "Blas": 2,  # "San Blas"
}


def build_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    """Return a :class:`requests.Session` with a UA header and retry/backoff."""
    session = requests.Session()
    session.headers["User-Agent"] = user_agent
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_driver(geckodriver_path: str | None, headless: bool) -> webdriver.Firefox:
    """Create a Firefox WebDriver."""
    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument("-headless")
    service = Service(executable_path=geckodriver_path) if geckodriver_path else Service()
    return webdriver.Firefox(service=service, options=options)


def accept_cookies(driver: webdriver.Firefox, timeout: float = 10) -> None:
    """Click the cookie-consent button if it shows up."""
    try:
        button = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, COOKIE_ACCEPT_XPATH))
        )
        button.click()
        time.sleep(3)
    except TimeoutException:
        logger.debug("No cookie banner to accept")


def scroll_to_bottom(driver: webdriver.Firefox, pause: float = SCROLL_PAUSE_TIME) -> None:
    """Scroll down one screen at a time so lazy-loaded cards render."""
    screen_height = driver.execute_script("return window.screen.height;")
    last_height = 0
    for i in range(1, MAX_SCROLLS + 1):
        driver.execute_script(f"window.scrollTo(0, {screen_height} * {i});")
        time.sleep(pause)
        scroll_height = driver.execute_script("return document.body.scrollHeight;")
        # Stop when we have scrolled past the bottom, or the page stopped growing.
        if screen_height * i > scroll_height or scroll_height == last_height:
            break
        last_height = scroll_height


def card_elements(driver: webdriver.Firefox) -> list:
    """Return the listing-card elements on the current results page."""
    return driver.find_elements(By.CLASS_NAME, CARD_LINK_CLASS)


def district_from_label(text: str, prefix: str = "") -> str | None:
    """Extract the district from a card's location label.

    The label reads ``<neighbourhood>, ... <district>``; the district is the last
    token, except for the multi-word names in :data:`MULTIWORD_DISTRICTS`.
    """
    tokens = text[len(prefix) :].split()
    if not tokens:
        return None
    last = tokens[-1]
    words = MULTIWORD_DISTRICTS.get(last, 1)
    return " ".join(tokens[-words:])


def district_for(card) -> str | None:
    """Read the district from a single card element."""
    try:
        label = card.find_element(By.XPATH, DISTRICT_RELATIVE_XPATH)
    except NoSuchElementException:
        return None

    prefix = ""
    with contextlib.suppress(NoSuchElementException):
        prefix = label.find_element(By.TAG_NAME, "span").text

    return district_from_label(label.text, prefix)


def load_scraped_urls(path: str) -> set[str]:
    """Return the set of listing URLs already present in ``path`` (empty if none)."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["URL"] for row in csv.DictReader(handle) if row.get("URL")}


def scrape_search(
    driver: webdriver.Firefox,
    session: requests.Session,
    pages: int,
    start_page: int,
    delay: float,
    skip_urls: set[str],
) -> Iterator[Home]:
    """Yield the parsed listing for every card across ``pages`` results pages.

    Listings whose detail page parsed to nothing are logged and skipped (a run
    that skips *most* listings means the selectors need updating).
    """
    empty = 0
    for page in range(start_page, start_page + pages):
        logger.info("Search page %d", page)
        driver.get(SEARCH_URL_TEMPLATE.format(page=page))
        time.sleep(PAGE_LOAD_PAUSE)
        accept_cookies(driver)
        scroll_to_bottom(driver)

        cards = card_elements(driver)
        logger.info("  %d listings", len(cards))
        for card in cards:
            link = card.get_attribute("href")
            if not link or link in skip_urls:
                continue
            district = district_for(card)
            if district is None:
                logger.debug("  no district for %s, skipping", link)
                continue
            time.sleep(delay)
            home = scrape_listing(session, link, district)
            if home is None:
                continue
            if home.is_empty():
                empty += 1
                continue
            yield home
    if empty:
        logger.warning("%d listing(s) parsed with no data - check the CSS/XPath selectors", empty)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pages", type=int, default=1, help="number of search pages to scrape")
    parser.add_argument("--start-page", type=int, default=1, help="first search page (1-based)")
    parser.add_argument(
        "--geckodriver",
        default=os.environ.get("GECKODRIVER_PATH"),
        help="path to the geckodriver binary (default: $GECKODRIVER_PATH or PATH)",
    )
    parser.add_argument("--output", default="buildings_information.csv", help="output CSV path")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to --output and skip listing URLs it already contains",
    )
    parser.add_argument("--headless", action="store_true", help="run Firefox headless")
    parser.add_argument("--delay", type=float, default=5.0, help="seconds to wait between listing requests")
    parser.add_argument("--log-level", default="INFO", help="logging level (default: INFO)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")

    skip_urls = load_scraped_urls(args.output) if args.resume else set()
    if skip_urls:
        logger.info("Resuming: %d listings already in %s", len(skip_urls), args.output)

    session = build_session()
    driver = build_driver(args.geckodriver, args.headless)

    mode = "a" if (args.resume and skip_urls) else "w"
    written = 0
    try:
        with open(args.output, mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
            if mode == "w":
                writer.writeheader()
            for home in scrape_search(driver, session, args.pages, args.start_page, args.delay, skip_urls):
                writer.writerow(home.to_csv_row())
                handle.flush()
                written += 1
    finally:
        driver.quit()

    logger.info("Wrote %d new rows to %s", written, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
