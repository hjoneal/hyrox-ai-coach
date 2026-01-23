"""
Feature engineering pipeline for Hyrox performance prediction.

This module provides utilities for:
- Extracting run-based features (consistency, acceleration, pacing)
- Extracting station-based features (grouped by type, fatigue indicators)
- Extracting overall pacing features

Note: Assumes time columns are already in seconds (integer format).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy import stats


# =============================================================================
# Run Feature Extraction
# =============================================================================

class RunFeatureExtractor:
    """Extract features from the 8 running segments."""

    @staticmethod
    def extract(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from 8 running segments.

        Args:
            df: DataFrame with run_1 through run_8 columns (in seconds)

        Returns:
            DataFrame with run features
        """
        run_cols = [f"run_{i}" for i in range(1, 9)]

        # Verify columns exist
        missing = [c for c in run_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing run columns: {missing}")

        runs = df[run_cols].values.astype(float)
        features = pd.DataFrame(index=df.index)

        # Basic statistics
        features["run_mean"] = np.nanmean(runs, axis=1)
        features["run_std"] = np.nanstd(runs, axis=1)
        features["run_cv"] = features["run_std"] / features["run_mean"]

        # Half splits (fatigue indicator)
        first_half = np.nanmean(runs[:, :4], axis=1)
        second_half = np.nanmean(runs[:, 4:], axis=1)
        features["run_first_half_avg"] = first_half
        features["run_second_half_avg"] = second_half

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            features["run_acceleration"] = (second_half - first_half) / first_half
            features["run_acceleration"] = features["run_acceleration"].replace([np.inf, -np.inf], np.nan)

        # Trend analysis (linear regression slope)
        x = np.arange(8)
        slopes = []
        for row in runs:
            if np.any(np.isnan(row)) or np.any(row == 0):
                slopes.append(np.nan)
            else:
                slope, _ = np.polyfit(x, row, 1)
                slopes.append(slope)
        features["run_trend_slope"] = slopes

        # Extremes
        features["run_slowest_idx"] = np.nanargmax(runs, axis=1) + 1
        features["run_fastest_idx"] = np.nanargmin(runs, axis=1) + 1
        features["run_range"] = np.nanmax(runs, axis=1) - np.nanmin(runs, axis=1)

        return features


# =============================================================================
# Station Feature Extraction
# =============================================================================

class StationFeatureExtractor:
    """Extract features from the 8 workout stations."""

    # Station groupings by fitness type
    STATION_NAMES = {
        1: "skierg",       # 1000m SkiErg
        2: "sled_push",    # 50m Sled Push
        3: "sled_pull",    # 50m Sled Pull
        4: "burpees",      # 80 Burpee Broad Jumps
        5: "row",          # 1000m Row
        6: "farmers",      # 200m Farmers Carry
        7: "lunges",       # 100m Sandbag Lunges
        8: "wall_balls",   # 100 Wall Balls
    }

    # Grouped by fitness component
    CARDIO_STATIONS = [1, 5]           # SkiErg, Row
    STRENGTH_STATIONS = [2, 3, 6]      # Sled Push, Sled Pull, Farmers
    ENDURANCE_STATIONS = [4, 7, 8]     # Burpees, Lunges, Wall Balls

    @staticmethod
    def extract(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from 8 workout stations.

        Args:
            df: DataFrame with station_1 through station_8 columns (in seconds)

        Returns:
            DataFrame with station features
        """
        station_cols = [f"station_{i}" for i in range(1, 9)]

        # Verify columns exist
        missing = [c for c in station_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing station columns: {missing}")

        stations = df[station_cols].values.astype(float)
        features = pd.DataFrame(index=df.index)

        # Basic statistics
        features["station_mean"] = np.nanmean(stations, axis=1)
        features["station_std"] = np.nanstd(stations, axis=1)
        features["station_cv"] = features["station_std"] / features["station_mean"]

        # Type-based groupings
        cardio_cols = [f"station_{i}" for i in StationFeatureExtractor.CARDIO_STATIONS]
        strength_cols = [f"station_{i}" for i in StationFeatureExtractor.STRENGTH_STATIONS]
        endurance_cols = [f"station_{i}" for i in StationFeatureExtractor.ENDURANCE_STATIONS]

        features["cardio_time"] = df[cardio_cols].sum(axis=1)
        features["strength_time"] = df[strength_cols].sum(axis=1)
        features["endurance_time"] = df[endurance_cols].sum(axis=1)

        total_station = np.nansum(stations, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            features["cardio_ratio"] = features["cardio_time"] / total_station
            features["strength_ratio"] = features["strength_time"] / total_station
            features["endurance_ratio"] = features["endurance_time"] / total_station

        # Fatigue indicators
        first_half = np.nanmean(stations[:, :4], axis=1)
        second_half = np.nanmean(stations[:, 4:], axis=1)
        features["station_first_half_avg"] = first_half
        features["station_second_half_avg"] = second_half

        with np.errstate(divide='ignore', invalid='ignore'):
            features["station_fatigue_index"] = second_half / first_half
            features["station_fatigue_index"] = features["station_fatigue_index"].replace([np.inf, -np.inf], np.nan)

        # Extremes
        features["worst_station_idx"] = np.nanargmax(stations, axis=1) + 1
        features["best_station_idx"] = np.nanargmin(stations, axis=1) + 1

        return features


# =============================================================================
# Pacing Feature Extraction
# =============================================================================

class PacingFeatureExtractor:
    """Extract overall pacing and performance features."""

    @staticmethod
    def extract(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract overall pacing and performance features.

        Args:
            df: DataFrame with time columns in seconds

        Returns:
            DataFrame with pacing features
        """
        features = pd.DataFrame(index=df.index)

        # Time ratios
        if "total_run" in df.columns and "total_stations" in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                features["run_station_ratio"] = df["total_run"] / df["total_stations"]
                features["run_station_ratio"] = features["run_station_ratio"].replace([np.inf, -np.inf], np.nan)

        # Roxzone (transition) time
        if "roxzone_time" in df.columns:
            features["roxzone"] = df["roxzone_time"]

            if "overall_time" in df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    features["roxzone_ratio"] = df["roxzone_time"] / df["overall_time"]

        # Quarter analysis
        first_q_cols = ["run_1", "station_1", "run_2", "station_2"]
        last_q_cols = ["run_7", "station_7", "run_8", "station_8"]

        if all(c in df.columns for c in first_q_cols + last_q_cols):
            features["first_quarter_time"] = df[first_q_cols].sum(axis=1)
            features["last_quarter_time"] = df[last_q_cols].sum(axis=1)

            if "overall_time" in df.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    features["first_quarter_pct"] = features["first_quarter_time"] / df["overall_time"]
                    features["last_quarter_pct"] = features["last_quarter_time"] / df["overall_time"]

        # Positive/negative split analysis
        run_cols = [f"run_{i}" for i in range(1, 9)]
        station_cols = [f"station_{i}" for i in range(1, 9)]

        if all(c in df.columns for c in run_cols + station_cols):
            first_half_runs = df[[f"run_{i}" for i in range(1, 5)]].sum(axis=1)
            first_half_stations = df[[f"station_{i}" for i in range(1, 5)]].sum(axis=1)
            second_half_runs = df[[f"run_{i}" for i in range(5, 9)]].sum(axis=1)
            second_half_stations = df[[f"station_{i}" for i in range(5, 9)]].sum(axis=1)

            first_half_total = first_half_runs + first_half_stations
            second_half_total = second_half_runs + second_half_stations

            features["positive_split"] = (second_half_total > first_half_total).astype(int)
            features["split_difference"] = second_half_total - first_half_total

        # Finish strength (last 2 runs vs first 2 runs)
        if all(c in df.columns for c in ["run_1", "run_2", "run_7", "run_8"]):
            last_2_runs = df[["run_7", "run_8"]].mean(axis=1)
            first_2_runs = df[["run_1", "run_2"]].mean(axis=1)

            with np.errstate(divide='ignore', invalid='ignore'):
                features["finish_strength"] = (last_2_runs - first_2_runs) / first_2_runs
                features["finish_strength"] = features["finish_strength"].replace([np.inf, -np.inf], np.nan)

        return features


# =============================================================================
# Main Feature Engineering Pipeline
# =============================================================================

class HyroxFeatureEngineer:
    """
    Main feature engineering pipeline for Hyrox finish time prediction.

    Combines run features, station features, and pacing features
    into a single transform pipeline.
    """

    def __init__(self):
        self.run_extractor = RunFeatureExtractor()
        self.station_extractor = StationFeatureExtractor()
        self.pacing_extractor = PacingFeatureExtractor()
        self._feature_names = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline.

        Args:
            df: DataFrame with time columns in seconds

        Returns:
            DataFrame with original columns plus engineered features
        """
        df_copy = df.copy()

        # Calculate race_time from splits if not present
        run_cols = [f"run_{i}" for i in range(1, 9)]
        station_cols = [f"station_{i}" for i in range(1, 9)]

        if all(c in df_copy.columns for c in run_cols + station_cols):
            df_copy["race_time"] = (
                df_copy[run_cols].sum(axis=1) +
                df_copy[station_cols].sum(axis=1)
            )

        # Extract feature groups
        run_features = self.run_extractor.extract(df_copy)
        station_features = self.station_extractor.extract(df_copy)
        pacing_features = self.pacing_extractor.extract(df_copy)

        # Store feature names
        self._feature_names = (
            list(run_features.columns) +
            list(station_features.columns) +
            list(pacing_features.columns)
        )

        # Combine
        feature_df = pd.concat([
            df_copy,
            run_features,
            station_features,
            pacing_features
        ], axis=1)

        return feature_df

    def get_feature_names(self) -> List[str]:
        """Return list of all engineered feature column names."""
        return self._feature_names

    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract target variable (race time in seconds).

        Args:
            df: DataFrame with overall_time or race_time column

        Returns:
            Series containing target values
        """
        # Prefer overall_time (from HTML), fallback to calculated race_time
        if "overall_time" in df.columns:
            return df["overall_time"]
        elif "race_time" in df.columns:
            return df["race_time"]
        else:
            raise ValueError("No time column found. Ensure data has 'overall_time' or 'race_time'.")


# =============================================================================
# Feature Validation Utilities
# =============================================================================

class FeatureValidator:
    """Validate feature quality for modeling."""

    @staticmethod
    def check_distributions(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Check feature distributions for modeling suitability.

        Args:
            df: DataFrame containing features
            feature_cols: List of feature column names to check

        Returns:
            DataFrame with distribution statistics
        """
        stats_list = []
        for col in feature_cols:
            if col not in df.columns:
                continue

            data = df[col].dropna()
            if len(data) == 0:
                continue

            stats_dict = {
                "feature": col,
                "mean": data.mean(),
                "std": data.std(),
                "min": data.min(),
                "max": data.max(),
                "skewness": stats.skew(data) if len(data) > 2 else np.nan,
                "missing_pct": df[col].isna().mean() * 100,
                "unique_count": data.nunique()
            }
            stats_list.append(stats_dict)

        return pd.DataFrame(stats_list)

    @staticmethod
    def check_target_correlation(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "overall_time"
    ) -> pd.DataFrame:
        """
        Check correlation between features and target.

        Args:
            df: DataFrame containing features and target
            feature_cols: List of feature column names
            target_col: Name of target column

        Returns:
            DataFrame with correlation statistics sorted by strength
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        correlations = []
        for col in feature_cols:
            if col == target_col or col not in df.columns:
                continue

            valid = df[[col, target_col]].dropna()
            if len(valid) < 10:
                continue

            pearson_r, pearson_p = stats.pearsonr(valid[col], valid[target_col])
            spearman_r, spearman_p = stats.spearmanr(valid[col], valid[target_col])

            correlations.append({
                "feature": col,
                "pearson_corr": pearson_r,
                "pearson_p": pearson_p,
                "spearman_corr": spearman_r,
                "spearman_p": spearman_p
            })

        result = pd.DataFrame(correlations)
        if not result.empty:
            result = result.sort_values("pearson_corr", key=abs, ascending=False)

        return result

    @staticmethod
    def check_multicollinearity(
        df: pd.DataFrame,
        feature_cols: List[str],
        threshold: float = 0.9
    ) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated feature pairs.

        Args:
            df: DataFrame containing features
            feature_cols: List of feature column names
            threshold: Correlation threshold for flagging

        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < 2:
            return []

        corr_matrix = df[valid_cols].corr()
        high_corr = []

        for i, col1 in enumerate(valid_cols):
            for col2 in valid_cols[i + 1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > threshold:
                    high_corr.append((col1, col2, corr))

        return high_corr
