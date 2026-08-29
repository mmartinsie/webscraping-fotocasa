"""Turn the raw scraper output into a model-ready dataset.

Reads ``buildings_information.csv`` (the CSV produced by ``webscraping/main.py``:
``Precio, Distrito, Tipo, Habitaciones, Aseos, Superficie, Planta, Parking, URL``)
and writes a cleaned CSV with the columns the model scripts consume:
``Precio, Precio_m2, Habitaciones, Aseos, Superficie, Parking, Colegios, Tipo,
Distrito``.

    python prepare_dataset.py webscraping/buildings_information.csv -o dataset.csv

``Colegios`` (number of nearby schools) is not something the scraper collects; it
came from an external join in the original thesis pipeline. Provide it with
``--schools schools.csv`` (columns ``Distrito, Colegios``); otherwise the column
is written empty and the model scripts fall back to filling it with 1.
"""

from __future__ import annotations

import argparse

import pandas as pd

RAW_NUMERIC = ["Precio", "Habitaciones", "Aseos", "Superficie", "Planta"]
OUTPUT_COLUMNS = [
    "Precio",
    "Precio_m2",
    "Habitaciones",
    "Aseos",
    "Superficie",
    "Parking",
    "Colegios",
    "Tipo",
    "Distrito",
]


def clean(
    raw: pd.DataFrame,
    min_price: float,
    max_price: float,
    min_size: float,
    max_size: float,
) -> pd.DataFrame:
    df = raw.copy()
    for column in RAW_NUMERIC:
        df[column] = pd.to_numeric(df.get(column), errors="coerce")

    parking = pd.to_numeric(df.get("Parking"), errors="coerce")
    df["Parking"] = parking.fillna(0).astype(int).clip(0, 1)

    df = df.dropna(subset=["Precio", "Superficie"])
    df = df[df["Precio"].between(min_price, max_price) & df["Superficie"].between(min_size, max_size)]

    df["Precio_m2"] = (df["Precio"] / df["Superficie"]).round().astype(int)

    # Keep the whole-number columns as integers (nullable, so blanks stay blank).
    for column in ["Precio", "Superficie", "Habitaciones", "Aseos", "Planta"]:
        df[column] = df[column].astype("Int64")

    df = df.drop_duplicates()
    return df


def attach_schools(df: pd.DataFrame, schools_csv: str | None) -> pd.DataFrame:
    if schools_csv is None:
        df["Colegios"] = pd.NA
        return df
    schools = pd.read_csv(schools_csv)
    if {"Distrito", "Colegios"} - set(schools.columns):
        raise SystemExit("--schools CSV must have columns: Distrito, Colegios")
    return df.merge(schools[["Distrito", "Colegios"]], on="Distrito", how="left")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="raw scraper CSV (buildings_information.csv)")
    parser.add_argument("-o", "--output", default="dataset.csv", help="cleaned CSV path")
    parser.add_argument("--schools", help="CSV with columns Distrito, Colegios to join")
    parser.add_argument("--min-price", type=float, default=20_000)
    parser.add_argument("--max-price", type=float, default=3_000_000)
    parser.add_argument("--min-size", type=float, default=20)
    parser.add_argument("--max-size", type=float, default=1_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = pd.read_csv(args.input, header=0, encoding="latin1")
    print(f"Read {len(raw):,} raw rows from {args.input}")

    df = clean(raw, args.min_price, args.max_price, args.min_size, args.max_size)
    df = attach_schools(df, args.schools)
    df = df.reindex(columns=OUTPUT_COLUMNS)

    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(df):,} cleaned rows to {args.output}")
    if args.schools is None:
        print("Note: 'Colegios' left empty (no --schools CSV given).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
