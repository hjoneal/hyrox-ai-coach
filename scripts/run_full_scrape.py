"""Run full scrape for all 25 events and generate feature dataset."""

import sys
sys.path.insert(0, "/Users/harryneal/Documents/personal_projects/hyrox-ai-coach")

from src.ingestion.scraper import HyroxScraper
from src.ingestion.events import ALL_EVENTS
from src.processing.features import HyroxFeatureEngineer, FeatureValidator

print(f"Starting full scrape for {len(ALL_EVENTS)} events:")
for i, e in enumerate(ALL_EVENTS, 1):
    print(f"  {i}. {e.name} (Season {e.season})")

# Initialize scraper
scraper = HyroxScraper(season=5)

# Run full scrape
print(f"\n{'='*60}")
print("SCRAPING DATA")
print("="*60)

df = scraper.scrape_multiple_events(
    events=ALL_EVENTS,
    output_dir="data/raw/events",
    save_intermediate=True
)

if df.empty:
    print("ERROR: No data scraped!")
    sys.exit(1)

print(f"\nTotal participants scraped: {len(df)}")
print(f"Events covered: {df['event_name'].nunique()}")

# Save combined raw data
raw_output = "data/raw/hyrox_combined.csv"
df.to_csv(raw_output, index=False)
print(f"Saved raw data to {raw_output}")

# Run feature engineering
print(f"\n{'='*60}")
print("FEATURE ENGINEERING")
print("="*60)

engineer = HyroxFeatureEngineer()
df_features = engineer.fit_transform(df)

print(f"Generated {len(engineer.get_feature_names())} features")

# Get target stats
target = engineer.get_target(df_features)
print(f"\nTarget (race_time_seconds) stats:")
print(f"  Count: {len(target)}")
print(f"  Mean: {target.mean():.0f} seconds ({target.mean()/60:.1f} minutes)")
print(f"  Std: {target.std():.0f} seconds")
print(f"  Min: {target.min():.0f} seconds ({target.min()/60:.1f} minutes)")
print(f"  Max: {target.max():.0f} seconds ({target.max()/60:.1f} minutes)")

# Validate features
print(f"\n{'='*60}")
print("FEATURE VALIDATION")
print("="*60)

validator = FeatureValidator()
feature_names = engineer.get_feature_names()

# Top correlations
correlations = validator.check_target_correlation(df_features, feature_names)
print("\nTop 10 Feature-Target Correlations:")
print(correlations.head(10)[['feature', 'pearson_corr', 'spearman_corr']].to_string(index=False))

# High multicollinearity
high_corr = validator.check_multicollinearity(df_features, feature_names)
print(f"\nHighly correlated feature pairs (>0.9): {len(high_corr)}")

# Save processed data
processed_output = "data/processed/hyrox_features.csv"
df_features.to_csv(processed_output, index=False)
print(f"\nSaved processed features to {processed_output}")

print(f"\n{'='*60}")
print("SCRAPING COMPLETE")
print("="*60)
print(f"Raw data: {raw_output} ({len(df)} rows)")
print(f"Features: {processed_output} ({len(df_features.columns)} columns)")
