"""Scrape Madrid property listings from Fotocasa into a CSV file.

Drives Firefox with Selenium to walk the search-results pages, then downloads and
parses each listing with :mod:`listing`. Run it from inside the ``webscraping``
directory:

    python main.py --pages 5 --output buildings_information.csv

A geckodriver binary is required (https://github.com/mozilla/geckodriver); pass
its path with ``--geckodriver`` or the ``GECKODRIVER_PATH`` environment variable.

The XPaths/classes below reflect Fotocasa's DOM at the time the thesis was
written and will need revisiting if the site layout has changed.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time

import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

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
DISTRICT_XPATH_TEMPLATE = (
    "/html/body/div[1]/div[3]/div/div[4]/div[2]/div[1]/main/div[3]/section/article[{index}]"
    "/div/div[2]/a/div[3]/h3"
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
SCROLL_PAUSE_TIME = 0.5
PAGE_LOAD_PAUSE = 5


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
    i = 1
    while True:
        driver.execute_script(f"window.scrollTo(0, {screen_height} * {i});")
        i += 1
        time.sleep(pause)
        scroll_height = driver.execute_script("return document.body.scrollHeight;")
        if screen_height * i > scroll_height:
            break


def listing_links(driver: webdriver.Firefox) -> list[str]:
    """Return the href of every listing card on the current results page."""
    cards = driver.find_elements(By.CLASS_NAME, CARD_LINK_CLASS)
    return [href for card in cards if (href := card.get_attribute("href"))]


def district_for(driver: webdriver.Firefox, index: int) -> str | None:
    """Read the district shown on the ``index``-th card (1-based).

    The label text is ``<neighbourhood>, ... <district>``; we take the trailing
    token, with special cases for the multi-word Madrid districts whose names do
    not fit that rule.
    """
    try:
        label = driver.find_element(
            By.XPATH, DISTRICT_XPATH_TEMPLATE.format(index=index)
        )
    except NoSuchElementException:
        return None

    prefix = ""
    try:
        prefix = label.find_element(By.TAG_NAME, "span").text
    except NoSuchElementException:
        pass

    tokens = label.text[len(prefix):].split()
    if not tokens:
        return None
    district = tokens[-1]
    if district == "Vallecas":
        district = " ".join(tokens[-3:])
    elif district in ("Lineal", "Blas"):
        district = " ".join(tokens[-2:])
    return district


def scrape_search(
    driver: webdriver.Firefox,
    session: requests.Session,
    pages: int,
    start_page: int,
    delay: float,
) -> list[Home]:
    """Walk ``pages`` search-result pages and return the parsed listings."""
    homes: list[Home] = []
    for page in range(start_page, start_page + pages):
        logger.info("Search page %d", page)
        driver.get(SEARCH_URL_TEMPLATE.format(page=page))
        time.sleep(PAGE_LOAD_PAUSE)
        accept_cookies(driver)
        scroll_to_bottom(driver)

        links = listing_links(driver)
        logger.info("  %d listings", len(links))
        for index, link in enumerate(links, start=1):
            district = district_for(driver, index)
            if district is None:
                logger.debug("  no district for card %d, skipping", index)
                continue
            time.sleep(delay)
            home = scrape_listing(session, link, district)
            if home is not None:
                homes.append(home)
    return homes


def write_csv(homes: list[Home], path: str) -> None:
    """Write ``homes`` to ``path`` as CSV (overwriting any existing file)."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for home in homes:
            writer.writerow(home.to_csv_row())
    logger.info("Wrote %d rows to %s", len(homes), path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pages", type=int, default=1, help="number of search pages to scrape")
    parser.add_argument("--start-page", type=int, default=1, help="first search page (1-based)")
    parser.add_argument(
        "--geckodriver",
        default=os.environ.get("GECKODRIVER_PATH"),
        help="path to the geckodriver binary (default: $GECKODRIVER_PATH or PATH)",
    )
    parser.add_argument(
        "--output", default="buildings_information.csv", help="output CSV path"
    )
    parser.add_argument("--headless", action="store_true", help="run Firefox headless")
    parser.add_argument(
        "--delay", type=float, default=5.0, help="seconds to wait between listing requests"
    )
    parser.add_argument("--log-level", default="INFO", help="logging level (default: INFO)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )

    session = requests.Session()
    session.headers["User-Agent"] = DEFAULT_USER_AGENT

    driver = build_driver(args.geckodriver, args.headless)
    try:
        homes = scrape_search(driver, session, args.pages, args.start_page, args.delay)
    finally:
        driver.quit()

    write_csv(homes, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
