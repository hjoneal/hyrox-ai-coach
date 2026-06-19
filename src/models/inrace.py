"""
In-race finish time prediction: baseline, models, and grouped validation.

Three components:

1. ProportionalBaseline — the domain-naive yardstick. At checkpoint k,
   predict finish = (work time so far) / (median fraction of the finish time
   that work-through-checkpoint-k represents in training data), per gender.
   Any model that can't beat this isn't learning anything beyond "you're
   about 26% done after run 3".

2. LightGBM quantile models — one triple (q05, q50, q95) per checkpoint.
   q50 is the point prediction; (q05, q95) form a nominal 90% interval.

3. Leave-event-out CV with conformalized quantile regression (CQR).
   Random row splits leak event effects (same-venue athletes share course
   conditions), so every fold holds out one whole event. Within each fold's
   training events, a few events are held out as a conformal calibration set
   — grouped, not row-sampled, so the calibration residuals reflect the
   event-to-event shift the test fold actually faces.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.processing.checkpoint_features import (
    CHECKPOINTS,
    TARGET_COL,
    build_checkpoint_features,
)

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "verbose": -1,
}


class ProportionalBaseline:
    """Per-gender proportional extrapolation from cumulative work time.

    fraction[g, k] = median over training athletes of gender g of
                     (sum of splits 1..k) / overall_time
    prediction at k = cum_time_k / fraction[g, k]

    The fraction is computed against overall_time (which includes roxzone),
    so unobserved transition time is absorbed by the denominator.
    """

    def __init__(self):
        self.fractions_: pd.DataFrame | None = None  # index gender, cols k

    def fit(self, df: pd.DataFrame) -> "ProportionalBaseline":
        cum = df[CHECKPOINTS].astype(float).cumsum(axis=1)
        frac = cum.div(df[TARGET_COL].astype(float), axis=0)
        frac["gender"] = df["gender"].values
        self.fractions_ = frac.groupby("gender", observed=True).median()
        return self

    def predict(self, df: pd.DataFrame, k: int) -> np.ndarray:
        if self.fractions_ is None:
            raise RuntimeError("Baseline not fitted")
        seg = CHECKPOINTS[k - 1]
        cum_k = df[CHECKPOINTS[:k]].astype(float).sum(axis=1)
        frac_k = df["gender"].map(self.fractions_[seg]).astype(float)
        return (cum_k / frac_k).values


def _fit_quantile_models(X: pd.DataFrame, y: pd.Series, quantiles, params) -> dict:
    models = {}
    for q in quantiles:
        m = LGBMRegressor(objective="quantile", alpha=q, **params)
        m.fit(X, y)
        models[q] = m
    return models


def _cqr_correction(y_calib, lo_calib, hi_calib, alpha: float) -> float:
    """Split-conformal correction for a (lo, hi) quantile interval pair."""
    scores = np.maximum(lo_calib - y_calib, y_calib - hi_calib)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class LeaveEventOutResult:
    predictions: pd.DataFrame
    calib_events: dict = field(default_factory=dict)  # fold event -> calib events


def run_leave_event_out(
    df: pd.DataFrame,
    quantiles=(0.05, 0.5, 0.95),
    alpha: float = 0.1,
    n_calib_events: int = 4,
    lgbm_params: dict | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> LeaveEventOutResult:
    """Full leave-event-out experiment across all 16 checkpoints.

    Returns one row per (athlete, checkpoint) with baseline, point, raw
    quantile, and CQR-adjusted predictions — everything downstream metrics
    and figures need, so the expensive loop runs once.
    """
    params = {**DEFAULT_LGBM_PARAMS, **(lgbm_params or {}), "random_state": seed}
    q_lo, q_mid, q_hi = sorted(quantiles)
    events = np.array(sorted(df["event_id"].unique()))
    rng = np.random.default_rng(seed)

    # Checkpoint features depend only on the row, not the fold — build once.
    features_by_k = {k: build_checkpoint_features(df, k) for k in range(1, len(CHECKPOINTS) + 1)}
    y_all = df[TARGET_COL].astype(float)

    records = []
    calib_map = {}
    for fold_idx, test_event in enumerate(events):
        train_events = events[events != test_event]
        calib_events = rng.choice(train_events, size=n_calib_events, replace=False)
        calib_map[test_event] = list(calib_events)

        is_test = (df["event_id"] == test_event).values
        is_calib = df["event_id"].isin(calib_events).values
        is_train = ~is_test & ~is_calib

        baseline = ProportionalBaseline().fit(df[is_train | is_calib])

        if verbose:
            print(
                f"[fold {fold_idx + 1:>2}/{len(events)}] test={test_event} "
                f"(n={is_test.sum()}), train={is_train.sum()}, calib={is_calib.sum()}",
                flush=True,
            )

        for k in range(1, len(CHECKPOINTS) + 1):
            X = features_by_k[k]
            models = _fit_quantile_models(X[is_train], y_all[is_train], (q_lo, q_mid, q_hi), params)

            lo_c = models[q_lo].predict(X[is_calib])
            hi_c = models[q_hi].predict(X[is_calib])
            correction = _cqr_correction(y_all[is_calib].values, lo_c, hi_c, alpha)

            lo_t = models[q_lo].predict(X[is_test])
            mid_t = models[q_mid].predict(X[is_test])
            hi_t = models[q_hi].predict(X[is_test])

            records.append(
                pd.DataFrame(
                    {
                        "event_id": test_event,
                        "row": np.flatnonzero(is_test),
                        "checkpoint": k,
                        "segment": CHECKPOINTS[k - 1],
                        "y_true": y_all[is_test].values,
                        "baseline_pred": baseline.predict(df[is_test], k),
                        "model_pred": mid_t,
                        "q_lo": lo_t,
                        "q_hi": hi_t,
                        "q_lo_cqr": lo_t - correction,
                        "q_hi_cqr": hi_t + correction,
                    }
                )
            )

    return LeaveEventOutResult(predictions=pd.concat(records, ignore_index=True), calib_events=calib_map)


def summarise_by_checkpoint(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-checkpoint metrics pooled over all LOEO folds (test rows only)."""
    p = predictions
    grouped = p.groupby(["checkpoint", "segment"], sort=True)
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "mae_baseline": np.mean(np.abs(g.baseline_pred - g.y_true)),
                "mae_model": np.mean(np.abs(g.model_pred - g.y_true)),
                "mape_model": np.median(np.abs(g.model_pred - g.y_true) / g.y_true) * 100,
                "coverage_raw": np.mean((g.y_true >= g.q_lo) & (g.y_true <= g.q_hi)),
                "coverage_cqr": np.mean((g.y_true >= g.q_lo_cqr) & (g.y_true <= g.q_hi_cqr)),
                "width_raw": np.median(g.q_hi - g.q_lo),
                "width_cqr": np.median(g.q_hi_cqr - g.q_lo_cqr),
            }
        ),
        include_groups=False,
    ).reset_index()
    out["improvement_pct"] = 100 * (1 - out["mae_model"] / out["mae_baseline"])
    return out


def summarise_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per (event, checkpoint) metrics — the spread across LOEO folds."""
    grouped = predictions.groupby(["event_id", "checkpoint"], sort=True)
    return grouped.apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "mae_baseline": np.mean(np.abs(g.baseline_pred - g.y_true)),
                "mae_model": np.mean(np.abs(g.model_pred - g.y_true)),
                "coverage_cqr": np.mean((g.y_true >= g.q_lo_cqr) & (g.y_true <= g.q_hi_cqr)),
                "width_cqr": np.median(g.q_hi_cqr - g.q_lo_cqr),
            }
        ),
        include_groups=False,
    ).reset_index()
