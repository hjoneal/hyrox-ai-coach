"""Light gender-label scrape: list pages only, men and women separately.

The original OPEN scrape ran ungendered, so the combined dataset has no
gender column (and its age_group/nationality fields silently failed to
parse from detail pages). The sex-filtered LIST pages carry name, finish
time, age group, and nationality at ~100 athletes per request, so a full
label pass over all 24 scraped events is ~430 requests instead of ~21k
detail-page requests.

Output: data/raw/labels/{event}_{M|W}.csv (per-event cache, resumable)
        data/raw/hyrox_gender_labels.csv (combined)

The labels are joined onto the detail-page dataset downstream (see
src/processing/cleaning.py) on (event_id, name, finish time).
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from src.ingestion.scraper import HyroxScraper
from src.ingestion.events import (
    SEASON_5_EVENTS, SEASON_6_EVENTS, Division, Gender,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_COMBINED = PROJECT_ROOT / "data/raw/hyrox_combined.csv"
LABELS_DIR = PROJECT_ROOT / "data/raw/labels"
OUTPUT = PROJECT_ROOT / "data/raw/hyrox_gender_labels.csv"


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Only label events actually present in the scraped dataset
    # (S5 Chicago and all S7/S8 events never made it into the data).
    scraped_ids = set(pd.read_csv(RAW_COMBINED, usecols=["event_id"])["event_id"])
    events = [
        e for e in SEASON_5_EVENTS + SEASON_6_EVENTS
        if f"{Division.OPEN.value}_{e.event_id}" in scraped_ids
    ]
    print(f"Labeling {len(events)} events present in {RAW_COMBINED.name}")

    scraper = HyroxScraper(season=5)
    frames = []

    for idx, event in enumerate(events, 1):
        if scraper.season != event.season:
            scraper.set_season(event.season)
        full_event_id = f"{Division.OPEN.value}_{event.event_id}"
        safe_name = event.name.replace(" ", "_").replace("/", "-")

        for gender in (Gender.MEN, Gender.WOMEN):
            cache_file = LABELS_DIR / f"{safe_name}_{gender.value}.csv"
            if cache_file.exists():
                df = pd.read_csv(cache_file)
                print(f"[{idx}/{len(events)}] {event.name} {gender.value}: "
                      f"cached ({len(df)})")
            else:
                df = scraper.scrape_event_labels(full_event_id, gender=gender)
                if df.empty:
                    print(f"[{idx}/{len(events)}] {event.name} {gender.value}: "
                          f"WARNING — no rows")
                    continue
                df["event_name"] = event.name
                df.to_csv(cache_file, index=False)
                print(f"[{idx}/{len(events)}] {event.name} {gender.value}: "
                      f"scraped {len(df)}")
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(combined)} labels to {OUTPUT}")
    print(combined["gender"].value_counts().to_string())


if __name__ == "__main__":
    main()
