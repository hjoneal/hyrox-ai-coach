"""Test script for the updated scraper with a few events."""

import sys
sys.path.insert(0, "/Users/harryneal/Documents/personal_projects/hyrox-ai-coach")

from src.ingestion.scraper import HyroxScraper
from src.ingestion.events import get_events, SEASON_5_EVENTS

# Test with just 2 events to validate the scraper works
test_events = SEASON_5_EVENTS[:2]  # First 2 Season 5 events

print(f"Testing scraper with {len(test_events)} events:")
for e in test_events:
    print(f"  - {e.name} (Season {e.season}, ID: {e.event_id})")

scraper = HyroxScraper(season=5)

# Scrape with limit of 1 page per event (100 participants max per event)
df = scraper.scrape_multiple_events(
    events=test_events,
    output_dir="data/raw/test_events",
    save_intermediate=True
)

if not df.empty:
    print(f"\nSuccess! Scraped {len(df)} total participants")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample data:")
    print(df[["name", "event_name", "total_time"]].head())

    # Save combined test data
    output_file = "data/raw/test_combined.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")
else:
    print("No data scraped. Check logs for errors.")
