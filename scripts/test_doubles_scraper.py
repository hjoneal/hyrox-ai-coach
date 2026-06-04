"""Test scraper for Mens Pro Doubles with 2 events.

Validates that the HDP division prefix with gender=M filter returns
valid data with 8 runs + 8 stations before running a full scrape.
"""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ingestion.scraper import HyroxScraper
from src.ingestion.events import SEASON_5_EVENTS, Division, Gender
from src.processing.features import HyroxFeatureEngineer

# Test with first 2 Season 5 events
test_events = SEASON_5_EVENTS[:2]

print(f"Testing Mens Pro Doubles scraper with {len(test_events)} events:")
for e in test_events:
    print(f"  - {e.name} (Season {e.season}, ID: {e.event_id})")

scraper = HyroxScraper(season=5)

# Scrape pro doubles with men's filter
df = scraper.scrape_multiple_events(
    events=test_events,
    division=Division.PRO_DOUBLES,
    gender=Gender.MEN,
    output_dir="data/raw/test_events",
    save_intermediate=True
)

if not df.empty:
    print(f"\nSuccess! Scraped {len(df)} total participants")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")

    # Verify split structure (8 runs + 8 stations)
    run_cols = [f"run_{i}" for i in range(1, 9)]
    station_cols = [f"station_{i}" for i in range(1, 9)]
    has_runs = all(c in df.columns for c in run_cols)
    has_stations = all(c in df.columns for c in station_cols)
    print(f"\nSplit structure: runs={has_runs}, stations={has_stations}")

    # Verify division metadata
    meta_cols = ['division_code', 'division_name', 'gender_filter']
    for col in meta_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].iloc[0]}")

    # Show time stats
    if 'overall_time' in df.columns:
        times = df['overall_time']
        print(f"\nOverall time stats:")
        print(f"  Count: {len(times)}")
        print(f"  Mean: {times.mean():.0f}s ({times.mean()/60:.1f} min)")
        print(f"  Min:  {times.min():.0f}s ({times.min()/60:.1f} min)")
        print(f"  Max:  {times.max():.0f}s ({times.max()/60:.1f} min)")

    # Test feature engineering
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING TEST")
    print(f"{'='*60}")

    engineer = HyroxFeatureEngineer()
    df_features = engineer.fit_transform(df)
    feature_names = engineer.get_feature_names()

    print(f"Generated {len(feature_names)} features")
    print(f"Total columns: {len(df_features.columns)}")

    # Verify metadata survives feature engineering
    for col in meta_cols:
        if col in df_features.columns:
            print(f"  {col} preserved: {df_features[col].iloc[0]}")

    # Save test output
    output_file = "data/raw/test_doubles_combined.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved raw to {output_file}")

    features_file = "data/processed/test_doubles_features.csv"
    df_features.to_csv(features_file, index=False)
    print(f"Saved features to {features_file}")

else:
    print("\nNo data scraped. This could mean:")
    print("  1. These events don't have Pro Doubles results")
    print("  2. Run scripts/discover_events.py first to find valid events")
