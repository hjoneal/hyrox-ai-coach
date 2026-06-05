"""Build the clean modeling table from the raw combined scrape + labels.

Applies src/processing/cleaning.py (rules R1-R6, logged with counts) and
writes data/processed/hyrox_clean.csv.
"""

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from src.processing.cleaning import clean_raw_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data/raw/hyrox_combined.csv"
LABELS = PROJECT_ROOT / "data/raw/hyrox_gender_labels.csv"
OUT = PROJECT_ROOT / "data/processed/hyrox_clean.csv"

df = pd.read_csv(RAW)
labels = pd.read_csv(LABELS) if LABELS.exists() else None
if labels is None:
    print(f"WARNING: {LABELS} not found — building without gender labels")

clean = clean_raw_dataset(df, labels=labels)
clean.to_csv(OUT, index=False)

print(f"\nWrote {len(clean)} rows x {len(clean.columns)} cols to {OUT}")
print(f"is_modeling_row: {clean['is_modeling_row'].sum()}")
if "gender" in clean.columns:
    print(clean["gender"].value_counts(dropna=False).to_string())
