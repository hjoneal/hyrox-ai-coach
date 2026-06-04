"""Run full scrape for Mens Pro Doubles and generate feature dataset."""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.ingestion.scraper import HyroxScraper
from src.ingestion.events import ALL_EVENTS, Division, Gender
from src.processing.features import HyroxFeatureEngineer, FeatureValidator

DIVISION = Division.PRO_DOUBLES
GENDER = Gender.MEN
OUTPUT_DIR = "data/raw/events"
RAW_OUTPUT = "data/raw/hyrox_mens_pro_doubles_combined.csv"
FEATURES_OUTPUT = "data/processed/hyrox_mens_pro_doubles_features.csv"

print(f"Starting Mens Pro Doubles scrape for {len(ALL_EVENTS)} events:")
for i, e in enumerate(ALL_EVENTS, 1):
    print(f"  {i}. {e.name} (Season {e.season})")

# Initialize scraper
scraper = HyroxScraper(season=5)

# Run full scrape
print(f"\n{'='*60}")
print(f"SCRAPING DATA ({DIVISION.name}, {GENDER.name})")
print("="*60)

df = scraper.scrape_multiple_events(
    events=ALL_EVENTS,
    division=DIVISION,
    gender=GENDER,
    output_dir=OUTPUT_DIR,
    save_intermediate=True
)

if df.empty:
    print("ERROR: No data scraped! Some events may not have Pro Doubles results.")
    print("Run scripts/discover_events.py to identify which events have data.")
    sys.exit(1)

print(f"\nTotal participants scraped: {len(df)}")
print(f"Events covered: {df['event_name'].nunique()}")

# Save combined raw data
df.to_csv(RAW_OUTPUT, index=False)
print(f"Saved raw data to {RAW_OUTPUT}")

# Show metadata columns
print(f"\nMetadata columns:")
meta_cols = ['division_code', 'division_name', 'gender_filter',
             'bib_number', 'age_group', 'nationality', 'rank_overall',
             'rank_age_group', 'bonus_time', 'penalty_time', 'roxzone_time',
             'best_run_lap']
for col in meta_cols:
    if col in df.columns:
        non_empty = df[col].notna().sum() if df[col].dtype == 'object' else (df[col] > 0).sum()
        print(f"  {col}: {non_empty} non-empty values")

# Run feature engineering
print(f"\n{'='*60}")
print("FEATURE ENGINEERING")
print("="*60)

engineer = HyroxFeatureEngineer()
df_features = engineer.fit_transform(df)

print(f"Generated {len(engineer.get_feature_names())} features")

# Get target stats
target = engineer.get_target(df_features)
print(f"\nTarget (overall_time) stats:")
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
if not correlations.empty:
    print("\nTop 10 Feature-Target Correlations:")
    print(correlations.head(10)[['feature', 'pearson_corr', 'spearman_corr']].to_string(index=False))

# Save processed data
df_features.to_csv(FEATURES_OUTPUT, index=False)
print(f"\nSaved processed features to {FEATURES_OUTPUT}")

print(f"\n{'='*60}")
print("SCRAPING COMPLETE")
print("="*60)
print(f"Raw data: {RAW_OUTPUT} ({len(df)} rows, {len(df.columns)} columns)")
print(f"Features: {FEATURES_OUTPUT} ({len(df_features.columns)} columns)")
