"""
Feature engineering pipeline for Hyrox performance prediction.

This module provides utilities for:
- Converting time strings to numeric seconds
- Extracting run-based features (consistency, acceleration, pacing)
- Extracting station-based features (grouped by type, fatigue indicators)
- Extracting overall pacing features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy import stats


# =============================================================================
# Time Conversion Utilities
# =============================================================================

def parse_timedelta_string(time_str: str) -> float:
    """
    Convert timedelta string '0 days 00:04:30' to seconds.

    Args:
        time_str: String in format '0 days HH:MM:SS' or 'HH:MM:SS'

    Returns:
        Total seconds as float, or NaN if invalid

    Examples:
        >>> parse_timedelta_string('0 days 00:04:30')
        270.0
        >>> parse_timedelta_string('0 days 01:15:30')
        4530.0
    """
    if pd.isna(time_str) or time_str == "" or time_str is None:
        return np.nan

    time_str = str(time_str).strip()

    # Handle "0 days HH:MM:SS" format
    if "days" in time_str:
        parts = time_str.split(" ")
        try:
            days = int(parts[0])
            time_part = parts[2] if len(parts) >= 3 else "00:00:00"
        except (ValueError, IndexError):
            return np.nan
    else:
        days = 0
        time_part = time_str

    try:
        time_components = time_part.split(":")
        if len(time_components) == 3:
            h, m, s = map(int, time_components)
        elif len(time_components) == 2:
            h = 0
            m, s = map(int, time_components)
        else:
            return np.nan

        return float(days * 86400 + h * 3600 + m * 60 + s)
    except (ValueError, AttributeError):
        return np.nan


def convert_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all time columns from string to seconds.

    Identifies columns containing 'run_', 'station_', 'total_', or 'time'
    and creates new columns with '_seconds' suffix.

    Args:
        df: DataFrame with time string columns

    Returns:
        DataFrame with additional _seconds columns
    """
    df_converted = df.copy()

    # Identify time columns
    time_cols = [
        c for c in df.columns
        if any(x in c.lower() for x in ["run_", "station_", "total_"])
        and "_seconds" not in c
    ]

    for col in time_cols:
        new_col = f"{col}_seconds"
        df_converted[new_col] = df[col].apply(parse_timedelta_string)

    # Calculate actual race time from splits (more reliable than total_time)
    run_cols = [f"run_{i}_seconds" for i in range(1, 9)]
    station_cols = [f"station_{i}_seconds" for i in range(1, 9)]

    if all(c in df_converted.columns for c in run_cols + station_cols):
        df_converted["race_time_seconds"] = (
            df_converted[run_cols].sum(axis=1) +
            df_converted[station_cols].sum(axis=1)
        )

    return df_converted


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
            df: DataFrame with run_1_seconds through run_8_seconds columns

        Returns:
            DataFrame with run features
        """
        run_cols = [f"run_{i}_seconds" for i in range(1, 9)]

        # Verify columns exist
        missing = [c for c in run_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing run columns: {missing}")

        runs = df[run_cols].values
        features = pd.DataFrame(index=df.index)

        # Basic statistics
        features["run_mean_seconds"] = np.nanmean(runs, axis=1)
        features["run_std_seconds"] = np.nanstd(runs, axis=1)
        features["run_cv"] = features["run_std_seconds"] / features["run_mean_seconds"]

        # Half splits (fatigue indicator)
        first_half = np.nanmean(runs[:, :4], axis=1)
        second_half = np.nanmean(runs[:, 4:], axis=1)
        features["run_first_half_avg"] = first_half
        features["run_second_half_avg"] = second_half
        features["run_acceleration"] = (second_half - first_half) / first_half

        # Trend analysis (linear regression slope)
        x = np.arange(8)
        slopes = []
        for row in runs:
            if np.any(np.isnan(row)):
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
            df: DataFrame with station_1_seconds through station_8_seconds columns

        Returns:
            DataFrame with station features
        """
        station_cols = [f"station_{i}_seconds" for i in range(1, 9)]

        # Verify columns exist
        missing = [c for c in station_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing station columns: {missing}")

        stations = df[station_cols].values
        features = pd.DataFrame(index=df.index)

        # Basic statistics
        features["station_mean_seconds"] = np.nanmean(stations, axis=1)
        features["station_std_seconds"] = np.nanstd(stations, axis=1)
        features["station_cv"] = (
            features["station_std_seconds"] / features["station_mean_seconds"]
        )

        # Type-based groupings
        cardio_cols = [f"station_{i}_seconds" for i in StationFeatureExtractor.CARDIO_STATIONS]
        strength_cols = [f"station_{i}_seconds" for i in StationFeatureExtractor.STRENGTH_STATIONS]
        endurance_cols = [f"station_{i}_seconds" for i in StationFeatureExtractor.ENDURANCE_STATIONS]

        features["cardio_time"] = df[cardio_cols].sum(axis=1)
        features["strength_time"] = df[strength_cols].sum(axis=1)
        features["endurance_time"] = df[endurance_cols].sum(axis=1)

        total_station = np.nansum(stations, axis=1)
        features["cardio_ratio"] = features["cardio_time"] / total_station
        features["strength_ratio"] = features["strength_time"] / total_station
        features["endurance_ratio"] = features["endurance_time"] / total_station

        # Fatigue indicators
        first_half = np.nanmean(stations[:, :4], axis=1)
        second_half = np.nanmean(stations[:, 4:], axis=1)
        features["station_first_half_avg"] = first_half
        features["station_second_half_avg"] = second_half
        features["station_fatigue_index"] = second_half / first_half

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
        if "total_run_seconds" in df.columns and "total_stations_seconds" in df.columns:
            features["run_station_ratio"] = (
                df["total_run_seconds"] / df["total_stations_seconds"]
            )

        # Roxzone (transition) time
        if "station_0_seconds" in df.columns:
            features["roxzone_seconds"] = df["station_0_seconds"]
        elif all(c in df.columns for c in ["total_time_seconds", "total_run_seconds", "total_stations_seconds"]):
            features["roxzone_seconds"] = (
                df["total_time_seconds"] -
                df["total_run_seconds"] -
                df["total_stations_seconds"]
            )

        if "roxzone_seconds" in features.columns and "total_time_seconds" in df.columns:
            features["roxzone_ratio"] = features["roxzone_seconds"] / df["total_time_seconds"]

        # Quarter analysis
        first_q_cols = [
            "run_1_seconds", "station_1_seconds",
            "run_2_seconds", "station_2_seconds"
        ]
        last_q_cols = [
            "run_7_seconds", "station_7_seconds",
            "run_8_seconds", "station_8_seconds"
        ]

        if all(c in df.columns for c in first_q_cols + last_q_cols):
            features["first_quarter_time"] = df[first_q_cols].sum(axis=1)
            features["last_quarter_time"] = df[last_q_cols].sum(axis=1)

            if "total_time_seconds" in df.columns:
                features["first_quarter_pct"] = (
                    features["first_quarter_time"] / df["total_time_seconds"]
                )
                features["last_quarter_pct"] = (
                    features["last_quarter_time"] / df["total_time_seconds"]
                )

        # Positive/negative split analysis
        run_cols = [f"run_{i}_seconds" for i in range(1, 9)]
        station_cols = [f"station_{i}_seconds" for i in range(1, 9)]

        if all(c in df.columns for c in run_cols + station_cols):
            first_half_runs = df[[f"run_{i}_seconds" for i in range(1, 5)]].sum(axis=1)
            first_half_stations = df[[f"station_{i}_seconds" for i in range(1, 5)]].sum(axis=1)
            second_half_runs = df[[f"run_{i}_seconds" for i in range(5, 9)]].sum(axis=1)
            second_half_stations = df[[f"station_{i}_seconds" for i in range(5, 9)]].sum(axis=1)

            first_half_total = first_half_runs + first_half_stations
            second_half_total = second_half_runs + second_half_stations

            features["positive_split"] = (second_half_total > first_half_total).astype(int)
            features["split_difference"] = second_half_total - first_half_total

        # Finish strength (last 2 runs vs first 2 runs)
        if all(c in df.columns for c in ["run_1_seconds", "run_2_seconds", "run_7_seconds", "run_8_seconds"]):
            last_2_runs = df[["run_7_seconds", "run_8_seconds"]].mean(axis=1)
            first_2_runs = df[["run_1_seconds", "run_2_seconds"]].mean(axis=1)
            features["finish_strength"] = (last_2_runs - first_2_runs) / first_2_runs

        return features


# =============================================================================
# Main Feature Engineering Pipeline
# =============================================================================

class HyroxFeatureEngineer:
    """
    Main feature engineering pipeline for Hyrox finish time prediction.

    Combines time conversion, run features, station features, and pacing features
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

        Steps:
        1. Convert time strings to seconds
        2. Extract run features
        3. Extract station features
        4. Extract pacing features
        5. Combine and return feature matrix

        Args:
            df: Raw DataFrame with time string columns

        Returns:
            DataFrame with original columns plus engineered features
        """
        # Step 1: Convert times
        df_converted = convert_time_columns(df)

        # Step 2-4: Extract feature groups
        run_features = self.run_extractor.extract(df_converted)
        station_features = self.station_extractor.extract(df_converted)
        pacing_features = self.pacing_extractor.extract(df_converted)

        # Store feature names
        self._feature_names = (
            list(run_features.columns) +
            list(station_features.columns) +
            list(pacing_features.columns)
        )

        # Step 5: Combine
        feature_df = pd.concat([
            df_converted,
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

        Uses race_time_seconds (calculated from splits) as the target,
        which is more reliable than total_time from the HTML.

        Args:
            df: DataFrame with race_time_seconds column

        Returns:
            Series containing target values
        """
        if "race_time_seconds" in df.columns:
            return df["race_time_seconds"]
        elif "total_time_seconds" in df.columns:
            return df["total_time_seconds"]
        else:
            raise ValueError("No time column found. Run fit_transform first.")


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
        target_col: str = "race_time_seconds"
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
        # Filter to existing columns
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
