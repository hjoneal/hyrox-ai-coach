"""Cleaning policy for the raw combined Hyrox dataset.

Every rule found during the Phase 1 data audit is implemented here as an
explicit, counted transformation — the clean table is reproducible from
the raw scrape and the audit trail prints with every build. Rules:

R1  Drop exact duplicate rows. The results site renders duplicate <li>
    rows for some athletes (desktop + mobile layouts), which the detail
    scraper followed twice — these are render artifacts, not re-entries.
R2  Recompute derived fields (total_run, total_stations, best_run_lap)
    from the 16 splits; impute roxzone_time = overall - sum(splits)
    where it is missing (zero). S6 Paris published no roxzone at all.
R3  Convert zero splits to NaN (zeros encode missing checkpoint mats,
    not zero-second segments) and record n_missing_splits.
R4  Flag rows whose splits do not reconcile with the official overall
    time (|overall - (sum(splits) + roxzone)| > CONSISTENCY_TOL_S).
R5  Join gender / age group / nationality labels from the sex-filtered
    list-page scrape (see scripts/scrape_gender_labels.py), exact-match
    on (event_id, name, finish time) with a (event_id, name) fallback
    where that pair is unique on both sides.
R6  Drop columns that are empty in every row (the detail-page parser
    silently failed on them) or zero in every row.

Rows are flagged, not dropped (except R1): `is_modeling_row` marks rows
with all 16 splits present and a reconciled overall time. Downstream
code chooses its own strictness.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RUN_COLS = [f"run_{i}" for i in range(1, 9)]
STATION_COLS = [f"station_{i}" for i in range(1, 9)]
SPLIT_COLS = RUN_COLS + STATION_COLS

# Empty (all-NaN) or all-zero columns from silent detail-page parse
# failures; age_group/nationality come back via the label join (R5).
DEAD_COLS = ["division", "age_group", "nationality", "disqual_reason",
             "bonus_time", "penalty_time"]

# Officially-published splits reconcile to the overall time within a few
# seconds (segment-level rounding); beyond a minute something is wrong
# with the row, not the rounding.
CONSISTENCY_TOL_S = 60


def clean_raw_dataset(
    df: pd.DataFrame,
    labels: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply the full cleaning policy to the raw combined dataset.

    Args:
        df: Raw combined scrape (schema of data/raw/hyrox_combined.csv)
        labels: Optional gender-label table
            (schema of data/raw/hyrox_gender_labels.csv)

    Returns:
        Cleaned DataFrame with audit flag columns.
    """
    df = df.copy()
    n0 = len(df)

    # --- R1: exact duplicates -------------------------------------------
    df = df.drop_duplicates(ignore_index=True)
    logger.info("R1: dropped %d exact duplicate rows (%d -> %d)",
                n0 - len(df), n0, len(df))

    # --- R2: recompute derived fields -----------------------------------
    splits_sum = df[SPLIT_COLS].where(df[SPLIT_COLS] > 0).sum(axis=1, min_count=16)
    all_splits_present = (df[SPLIT_COLS] > 0).all(axis=1)

    n_rox_imputed = 0
    rox_missing = (df["roxzone_time"] == 0) & all_splits_present
    implied_rox = df["overall_time"] - splits_sum
    # Only impute a plausible transition total (non-negative, < 1 hour)
    imputable = rox_missing & implied_rox.between(0, 3600)
    df.loc[imputable, "roxzone_time"] = implied_rox[imputable]
    df["roxzone_imputed"] = imputable
    n_rox_imputed = int(imputable.sum())
    logger.info("R2: imputed roxzone_time for %d rows (was 0, splits complete)",
                n_rox_imputed)

    runs = df[RUN_COLS].where(df[RUN_COLS] > 0)
    df["total_run"] = runs.sum(axis=1, min_count=8)
    df["total_stations"] = df[STATION_COLS].where(df[STATION_COLS] > 0).sum(
        axis=1, min_count=8)
    df["best_run_lap"] = runs.min(axis=1)
    logger.info("R2: recomputed total_run / total_stations / best_run_lap from splits")

    # --- R3: zero splits -> NaN ------------------------------------------
    n_zero_rows = int((df[SPLIT_COLS] == 0).any(axis=1).sum())
    df["n_missing_splits"] = (df[SPLIT_COLS] == 0).sum(axis=1)
    df[SPLIT_COLS] = df[SPLIT_COLS].where(df[SPLIT_COLS] > 0)
    rox = df["roxzone_time"].where(df["roxzone_time"] > 0)
    df["roxzone_time"] = rox
    logger.info("R3: %d rows had >=1 zero split -> NaN (n_missing_splits set)",
                n_zero_rows)

    # --- R4: consistency flag --------------------------------------------
    recon = df[SPLIT_COLS].sum(axis=1, min_count=16) + df["roxzone_time"]
    df["splits_consistent"] = (
        (df["overall_time"] - recon).abs() <= CONSISTENCY_TOL_S
    ).fillna(False)
    logger.info("R4: %d rows fail consistency (|overall - splits - rox| > %ds)",
                int((~df["splits_consistent"]).sum()), CONSISTENCY_TOL_S)

    # --- R5: gender label join -------------------------------------------
    if labels is not None:
        df = _join_gender_labels(df, labels)

    # --- R6: dead columns --------------------------------------------------
    drop = [c for c in DEAD_COLS if c in df.columns]
    df = df.drop(columns=drop)
    logger.info("R6: dropped dead columns: %s", drop)

    # --- final modeling flag -----------------------------------------------
    df["is_modeling_row"] = (
        (df["n_missing_splits"] == 0) & df["splits_consistent"]
    )
    logger.info("Final: %d / %d rows flagged is_modeling_row",
                int(df["is_modeling_row"].sum()), len(df))

    return df


def _join_gender_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """R5: attach gender / age_group / nationality from list-page labels.

    Exact match on (event_id, name, finish time) first; fall back to
    (event_id, name) where that pair is unique on both sides. Unmatched
    rows keep NaN gender.
    """
    lab = labels.copy()
    lab_cols = ["gender", "age_group", "nationality", "rank_gender"]

    # A label key that maps to conflicting genders is unusable
    key3 = ["event_id", "name", "finish_time"]
    lab3 = (lab.drop_duplicates(subset=key3 + ["gender"])
               .groupby(key3, as_index=False)
               .filter(lambda g: g["gender"].nunique() == 1)
               .drop_duplicates(subset=key3))

    merged = df.merge(
        lab3[key3 + lab_cols].rename(columns={"finish_time": "overall_time"}),
        on=["event_id", "name", "overall_time"],
        how="left",
    )

    # Fallback: (event_id, name) unique on both sides
    unmatched = merged["gender"].isna()
    key2 = ["event_id", "name"]
    lab2 = lab.drop_duplicates(subset=key2, keep=False)
    df2_unique = ~merged.duplicated(subset=key2, keep=False)
    fallback_idx = merged.index[unmatched & df2_unique]
    if len(fallback_idx):
        fb = (merged.loc[fallback_idx, key2]
                    .merge(lab2[key2 + lab_cols], on=key2, how="left")
                    .set_index(fallback_idx))
        for c in lab_cols:
            merged.loc[fallback_idx, c] = fb[c]

    n_exact = int((~unmatched).sum())
    n_fb = int(merged.loc[fallback_idx, "gender"].notna().sum()) if len(fallback_idx) else 0
    n_miss = int(merged["gender"].isna().sum())
    logger.info("R5: gender labels — %d exact, %d fallback, %d unmatched (%.2f%% labeled)",
                n_exact, n_fb, n_miss, 100 * (1 - n_miss / len(merged)))

    return merged
