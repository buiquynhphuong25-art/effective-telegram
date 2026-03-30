# Lending Club Credit Risk Pipeline

End-to-end machine learning pipeline for **credit default prediction** and **interest rate optimisation** on the Lending Club loan dataset.

---

## Project Overview

| # | Module | Description |
|---|--------|-------------|
| 1 | `step1_preprocessing.py` | Data profiling, missing-value imputation, outlier removal, cube-root transformation |
| 2 | `step2_feature_engineering.py` | Feature creation, one-hot encoding, train/test split, StandardScaler |
| 3 | `step3_model_training.py` | LightGBM, XGBoost, Logistic Regression — full & top-15 features, comparison table |
| 4 | `step4_tuning_calibration.py` | Optuna hyperparameter search, Platt sigmoid calibration, overfit check |
| 5 | `step5_simulation_optimization.py` | Four business scenarios (see below) |
| — | `run_pipeline.py` | Single-command orchestrator for all steps |

### Business Scenarios (Step 5)

| Scenario | Business Question | Method |
|----------|-------------------|--------|
| 1 | What is the optimal interest rate for this borrower? | Monte Carlo over rate grid, endogenous PD |
| 2 | What rate floor should each Sub_Grade (A1–G5) receive? | Median-profile + analytic E[Profit] scan |
| 3 | Which loans should we approve to maximise portfolio return? | PD-threshold scan, profit-ratio objective |
| 4 | How do we allocate capital across risk segments? | Greedy allocation by profit/loan ratio |

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/lending-club-pipeline.git
cd lending-club-pipeline

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the raw dataset
#    Download from Kaggle: "Lending Club Loan Data"
#    and place it as:  data/lending_club_loan.csv
mkdir -p data models
```

---

## Running the Pipeline

### Full pipeline (single command)

```bash
python run_pipeline.py --input data/lending_club_loan.csv
```

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to `lending_club_loan.csv` |
| `--n_trials` | `20` | Number of Optuna tuning trials |
| `--sample_idx` | `0` | Test-set borrower index for Scenario 1 |
| `--skip` | `""` | Comma-separated steps to skip, e.g. `"1,2"` |

```bash
# Skip steps 1 & 2 if data is already cleaned
python run_pipeline.py --input data/lending_club_loan.csv --skip 1,2 --n_trials 50
```

### Running steps individually

```bash
python step1_preprocessing.py       --input data/lending_club_loan.csv
python step2_feature_engineering.py --input data/lending_club_loan_cleaned.csv
python step3_model_training.py
python step4_tuning_calibration.py  --n_trials 20
python step5_simulation_optimization.py --sample_idx 0
```

---

## Output Files

```
data/
├── lending_club_loan_cleaned.csv   # Cleaned dataset (after step 1)
├── X_train.parquet                 # Train features
├── X_test.parquet                  # Test features
├── y_train.parquet                 # Train labels
├── y_test.parquet                  # Test labels
├── df_final.parquet                # Pre-encoding df (for sub_grade lookup)
├── result_with_pd.csv              # Test-set PD scores + actuals
├── kb2_pricing.csv                 # Pricing policy by sub_grade
├── kb3_optimization.csv            # Approval threshold scan results
├── kb4_allocation.csv              # Capital allocation table
├── subgrade_profile.csv            # Sub_grade risk/return profile
├── model_comparison.csv            # Cross-model comparison table
└── plots/                          # All visualisation PNGs

models/
├── final_model.pkl                 # Tuned LightGBM
├── cal_model.pkl                   # Platt-calibrated model
├── scaler.pkl                      # StandardScaler
├── selected_features.pkl           # Feature list
├── best_params.json                # Optuna best hyperparameters
└── best_model_pretune.pkl          # Pre-tuning best model
```

---

## Key Modelling Decisions

| Decision | Rationale |
|----------|-----------|
| Drop `installment` | Correlation ≈ 0.95 with `loan_amnt` — redundant |
| Drop `emp_title` | 173k unique values — no predictive lift |
| Drop `grade` | Data leakage — Lending Club sets grade after knowing int_rate |
| Cube-root transform | Normalises right-skewed variables without losing zeros |
| `mort_acc` imputation | Filled by `total_acc` group mean (highest correlation) |
| `is_unbalance=True` (LGBM) | Handles ~80/20 class imbalance without oversampling |
| Calibration (Platt) | Raw model is overconfident — sigmoid scaling corrects PD |
| Optuna (Bayesian search) | More efficient than grid search for 7 hyperparameters |
| F2-Score as secondary metric | Penalises false negatives (missed defaults) twice as much |

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
optuna>=3.3.0
joblib>=1.3.0
```

Python 3.9+ recommended.
