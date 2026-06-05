"""Test feature engineering pipeline on a sample of the real scraped data."""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
from src.processing.features import HyroxFeatureEngineer, FeatureValidator

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_COMBINED = PROJECT_ROOT / "data/raw/hyrox_combined.csv"

# Load a sample of the combined dataset (all times already in seconds)
df = pd.read_csv(RAW_COMBINED, nrows=2000)
print(f"Loaded {len(df)} rows from {RAW_COMBINED.name}")
print(f"Original columns: {len(df.columns)}")

# Run feature engineering pipeline
engineer = HyroxFeatureEngineer()
df_features = engineer.fit_transform(df)

print(f"\nAfter feature engineering:")
print(f"Total columns: {len(df_features.columns)}")
print(f"Engineered features: {len(engineer.get_feature_names())}")
print(f"\nEngineered feature names:")
for name in engineer.get_feature_names():
    print(f"  - {name}")

# Get target
target = engineer.get_target(df_features)
print(f"\nTarget (overall_time) stats:")
print(f"  Mean: {target.mean():.0f} seconds ({target.mean()/60:.1f} minutes)")
print(f"  Min: {target.min():.0f} seconds ({target.min()/60:.1f} minutes)")
print(f"  Max: {target.max():.0f} seconds ({target.max()/60:.1f} minutes)")

# Validate features
validator = FeatureValidator()
feature_names = engineer.get_feature_names()

print("\n" + "=" * 60)
print("Feature Distribution Analysis")
print("=" * 60)
dist_stats = validator.check_distributions(df_features, feature_names)
print(dist_stats.to_string(index=False))

print("\n" + "=" * 60)
print("Feature-Target Correlations (Top 15)")
print("=" * 60)
correlations = validator.check_target_correlation(df_features, feature_names)
print(correlations.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("Highly Correlated Feature Pairs (>0.9)")
print("=" * 60)
high_corr = validator.check_multicollinearity(df_features, feature_names)
for feat1, feat2, corr in high_corr:
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")

print("\nFeature pipeline OK")
