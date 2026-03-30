"""
run_pipeline.py — Full Pipeline Orchestrator
============================================
Runs all five steps end-to-end in order.

Usage
-----
  python run_pipeline.py --input /path/to/lending_club_loan.csv

Optional flags
--------------
  --n_trials  INT   Number of Optuna trials in step 4 (default: 20)
  --sample_idx INT  Borrower index for Scenario 1 in step 5 (default: 0)
  --skip      STR   Comma-separated step numbers to skip, e.g. "1,2"
                    (useful when raw data is already cleaned)

Step map
--------
  1  → step1_preprocessing.py       Data profiling & cleaning
  2  → step2_feature_engineering.py Feature engineering, encoding, split, scale
  3  → step3_model_training.py      LightGBM / XGBoost / LR comparison
  4  → step4_tuning_calibration.py  Optuna + Platt calibration
  5  → step5_simulation_optimization.py  Business simulations
"""

import argparse
import sys
import time
import step1_preprocessing       as s1
import step2_feature_engineering as s2
import step3_model_training      as s3
import step4_tuning_calibration  as s4
import step5_simulation_optimization as s5


def banner(step_no: int, title: str) -> None:
    line = "─" * 70
    print(f"\n{line}")
    print(f"  STEP {step_no} — {title}")
    print(f"{line}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lending Club BA Pipeline — Full Orchestrator"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to raw lending_club_loan.csv"
    )
    parser.add_argument(
        "--n_trials", type=int, default=20,
        help="Number of Optuna trials for step 4 (default: 20)"
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0,
        help="Test-set borrower index for Scenario 1 in step 5 (default: 0)"
    )
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated step numbers to skip, e.g. '1,2'"
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    skip   = {int(s) for s in args.skip.split(",") if s.strip().isdigit()}
    timings = {}

    # ── Step 1 ───────────────────────────────────────────────────────────────
    if 1 not in skip:
        banner(1, "Data Profiling & Preprocessing")
        t0 = time.time()
        s1.run(args.input)
        timings[1] = time.time() - t0
    else:
        print("[SKIP] Step 1 — using existing data/lending_club_loan_cleaned.csv")

    # ── Step 2 ───────────────────────────────────────────────────────────────
    if 2 not in skip:
        banner(2, "Feature Engineering, Encoding & Train/Test Split")
        t0 = time.time()
        s2.run("data/lending_club_loan_cleaned.csv")
        timings[2] = time.time() - t0
    else:
        print("[SKIP] Step 2 — using existing data/X_train.parquet …")

    # ── Step 3 ───────────────────────────────────────────────────────────────
    if 3 not in skip:
        banner(3, "Model Training & Comparison")
        t0 = time.time()
        s3.run()
        timings[3] = time.time() - t0
    else:
        print("[SKIP] Step 3 — using existing models/best_model_pretune.pkl")

    # ── Step 4 ───────────────────────────────────────────────────────────────
    if 4 not in skip:
        banner(4, "Hyperparameter Tuning & Calibration")
        t0 = time.time()
        s4.run(n_trials=args.n_trials)
        timings[4] = time.time() - t0
    else:
        print("[SKIP] Step 4 — using existing models/final_model.pkl & cal_model.pkl")

    # ── Step 5 ───────────────────────────────────────────────────────────────
    if 5 not in skip:
        banner(5, "Business Simulation & Optimisation")
        t0 = time.time()
        s5.run(sample_idx=args.sample_idx)
        timings[5] = time.time() - t0
    else:
        print("[SKIP] Step 5")

    # ── Summary ──────────────────────────────────────────────────────────────
    total = sum(timings.values())
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    for step, secs in timings.items():
        print(f"  Step {step}: {secs/60:.1f} min")
    print(f"  Total  : {total/60:.1f} min")
    print("\nKey output files:")
    print("  data/lending_club_loan_cleaned.csv   — cleaned dataset")
    print("  data/result_with_pd.csv              — test-set PD scores")
    print("  data/kb2_pricing.csv                 — pricing policy table")
    print("  data/kb3_optimization.csv            — approval threshold scan")
    print("  data/kb4_allocation.csv              — capital allocation")
    print("  models/final_model.pkl               — tuned LightGBM")
    print("  models/cal_model.pkl                 — calibrated model")
    print("  data/plots/                          — all charts")


if __name__ == "__main__":
    main()
