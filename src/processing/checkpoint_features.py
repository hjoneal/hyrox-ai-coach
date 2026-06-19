"""
Per-checkpoint feature construction for in-race finish time prediction.

A Hyrox race is a fixed sequence of 16 timed segments (8 runs and 8 stations,
alternating). After each segment the athlete crosses a timing mat — a
"checkpoint". The in-race prediction task is: standing at checkpoint k,
predict the finish time using only what is knowable at that moment.

Leakage rule: a feature is admissible at checkpoint k if and only if it can
be computed from segments 1..k plus pre-race static covariates (gender,
age group). Anything touching segments k+1..16 — or whole-race aggregates
like total roxzone time — is illegal.

Note on elapsed time: the cumulative sum of splits 1..k is NOT the athlete's
true clock time at checkpoint k, because transition (roxzone) time accrues
between segments and is only published as a race total. Models therefore see
"sum of work segments so far"; the gap to true elapsed time is absorbed by
the model / baseline mapping to overall_time (which includes roxzone).
"""

import numpy as np
import pandas as pd

# The 16 checkpoints in race order. CHECKPOINTS[k-1] is the segment whose
# completion defines checkpoint k.
CHECKPOINTS: list = []
for _i in range(1, 9):
    CHECKPOINTS.append(f"run_{_i}")
    CHECKPOINTS.append(f"station_{_i}")

STATIC_COLS = ["gender", "age_group"]
TARGET_COL = "overall_time"


def checkpoint_label(k: int) -> str:
    """Human-readable label for checkpoint k (1-based), e.g. 'after run 3'."""
    seg = CHECKPOINTS[k - 1]
    return "after " + seg.replace("_", " ")


def _slope_so_far(values: np.ndarray) -> np.ndarray:
    """Vectorised least-squares slope across columns for each row.

    Returns NaN when fewer than 2 columns are supplied.
    """
    n = values.shape[1]
    if n < 2:
        return np.full(values.shape[0], np.nan)
    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    y_centered = values - values.mean(axis=1, keepdims=True)
    return (y_centered * x_centered).sum(axis=1) / (x_centered ** 2).sum()


def build_checkpoint_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Build the admissible feature set at checkpoint k (1-based, 1..16).

    Args:
        df: clean modeling table with run_/station_ split columns (seconds),
            plus 'gender' and 'age_group' static covariates.
        k: checkpoint index — features use segments 1..k only.

    Returns:
        DataFrame of features, index-aligned with df. Gender and age_group
        are returned as pandas 'category' dtype (LightGBM-native handling).
    """
    if not 1 <= k <= len(CHECKPOINTS):
        raise ValueError(f"k must be in 1..{len(CHECKPOINTS)}, got {k}")

    legal = CHECKPOINTS[:k]
    run_cols = [c for c in legal if c.startswith("run_")]
    station_cols = [c for c in legal if c.startswith("station_")]

    feats = df[legal].astype(float).copy()
    feats["cum_time"] = feats[legal].sum(axis=1)

    runs = df[run_cols].astype(float).values
    feats["cum_run_time"] = runs.sum(axis=1)
    feats["run_mean_so_far"] = runs.mean(axis=1)
    feats["run_last"] = runs[:, -1]
    if len(run_cols) >= 2:
        feats["run_slope_so_far"] = _slope_so_far(runs)
        feats["run_last_vs_first"] = runs[:, -1] - runs[:, 0]
        feats["run_std_so_far"] = runs.std(axis=1)

    if station_cols:
        stations = df[station_cols].astype(float).values
        feats["cum_station_time"] = stations.sum(axis=1)
        feats["station_mean_so_far"] = stations.mean(axis=1)
        feats["station_last"] = stations[:, -1]
        feats["station_share_so_far"] = feats["cum_station_time"] / feats["cum_time"]

    for col in STATIC_COLS:
        feats[col] = df[col].astype("category")

    return feats


def prepare_modeling_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Filter the clean table to rows usable for in-race modeling.

    Keeps is_modeling_row rows with a known gender, and normalises the
    label-join column names (age_group_y -> age_group, etc.).
    """
    df = clean[clean["is_modeling_row"]].copy()
    df = df[df["gender"].notna()]
    rename = {}
    if "age_group_y" in df.columns:
        rename["age_group_y"] = "age_group"
    if "nationality_y" in df.columns:
        rename["nationality_y"] = "nationality"
    df = df.rename(columns=rename)
    df = df.drop(columns=[c for c in ("age_group_x", "nationality_x") if c in df.columns])
    return df.reset_index(drop=True)
