"""
Phase 2 experiment: in-race finish time prediction under leave-event-out CV.

Reads the clean modeling table, runs the full LOEO experiment (24 folds x
16 checkpoints x 3 quantile models + baseline + CQR), and writes:

    data/processed/phase2/predictions.csv.gz   one row per (athlete, checkpoint)
    data/processed/phase2/metrics_by_checkpoint.csv
    data/processed/phase2/metrics_by_fold.csv
    data/processed/phase2/calibration_events.csv

Report notebook 2 reads these artifacts rather than re-running the loop.
"""

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inrace import (  # noqa: E402
    run_leave_event_out,
    summarise_by_checkpoint,
    summarise_by_fold,
)
from src.processing.checkpoint_features import prepare_modeling_frame  # noqa: E402


def main():
    clean = pd.read_csv(PROJECT_ROOT / "data/processed/hyrox_clean.csv")
    df = prepare_modeling_frame(clean)
    print(f"Modeling frame: {len(df)} rows, {df.event_id.nunique()} events")

    start = time.time()
    result = run_leave_event_out(df)
    print(f"LOEO experiment finished in {time.time() - start:.0f}s")

    out_dir = PROJECT_ROOT / "data/processed/phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = result.predictions
    preds.to_csv(out_dir / "predictions.csv.gz", index=False, compression="gzip")

    by_cp = summarise_by_checkpoint(preds)
    by_cp.to_csv(out_dir / "metrics_by_checkpoint.csv", index=False)

    by_fold = summarise_by_fold(preds)
    by_fold.to_csv(out_dir / "metrics_by_fold.csv", index=False)

    calib = pd.DataFrame(
        [(e, c) for e, cal in result.calib_events.items() for c in cal],
        columns=["test_event", "calib_event"],
    )
    calib.to_csv(out_dir / "calibration_events.csv", index=False)

    pd.set_option("display.width", 160)
    print("\nPer-checkpoint summary (pooled over folds):")
    cols = ["segment", "mae_baseline", "mae_model", "improvement_pct",
            "coverage_raw", "coverage_cqr", "width_cqr"]
    print(by_cp[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
