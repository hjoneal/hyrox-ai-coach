"""Test feature engineering pipeline on scraped data."""

import sys
sys.path.insert(0, "/Users/harryneal/Documents/personal_projects/hyrox-ai-coach")

import pandas as pd
from src.processing.features import (
    HyroxFeatureEngineer,
    FeatureValidator,
    parse_timedelta_string
)

# Load test data
df = pd.read_csv("data/raw/test_combined.csv")
print(f"Loaded {len(df)} rows from test data")
print(f"Original columns: {len(df.columns)}")

# Test time parsing
sample_time = df['total_time'].iloc[0]
print(f"\nSample time string: {sample_time}")
print(f"Parsed to seconds: {parse_timedelta_string(sample_time)}")

# Run feature engineering pipeline
engineer = HyroxFeatureEngineer()
df_features = engineer.fit_transform(df)

print(f"\nAfter feature engineering:")
print(f"Total columns: {len(df_features.columns)}")
print(f"Engineered features: {len(engineer.get_feature_names())}")
print(f"\nEngineered feature names:")
for name in engineer.get_feature_names():
    print(f"  - {name}")

# Get target (race_time_seconds calculated from splits)
target = engineer.get_target(df_features)
print(f"\nTarget (race_time_seconds) stats:")
print(f"  Mean: {target.mean():.0f} seconds ({target.mean()/60:.1f} minutes)")
print(f"  Min: {target.min():.0f} seconds ({target.min()/60:.1f} minutes)")
print(f"  Max: {target.max():.0f} seconds ({target.max()/60:.1f} minutes)")

# Validate features
validator = FeatureValidator()
feature_names = engineer.get_feature_names()

print("\n" + "="*60)
print("Feature Distribution Analysis")
print("="*60)
dist_stats = validator.check_distributions(df_features, feature_names)
print(dist_stats.to_string(index=False))

print("\n" + "="*60)
print("Feature-Target Correlations (Top 15)")
print("="*60)
correlations = validator.check_target_correlation(df_features, feature_names)
print(correlations.head(15).to_string(index=False))

print("\n" + "="*60)
print("Highly Correlated Feature Pairs (>0.9)")
print("="*60)
high_corr = validator.check_multicollinearity(df_features, feature_names)
for feat1, feat2, corr in high_corr:
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")

# Save processed features
output_file = "data/processed/test_features.csv"
df_features.to_csv(output_file, index=False)
print(f"\nSaved processed features to {output_file}")
