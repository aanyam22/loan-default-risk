# Loan Default Risk — Supervised Classification with Fairness Evaluation

**Portfolio Project — Supervised Learning Methods**
**Author:** Adwoa Nyame

---

## Overview

This project builds a loan default prediction system for a simulated engagement with a consumer lending institution. The goal is a model that is not only accurate but defensible to regulators — one that produces well-calibrated probability estimates, supports threshold selection tied to business risk tolerance, and can be audited for disparate impact across borrower income groups.

Three classifiers are evaluated under a common framework:

| Model | ROC-AUC | PR-AUC | Brier Score | ECE |
|---|---|---|---|---|
| Logistic Regression | 0.6958 | 0.3607 | 0.1557 | 0.0320 |
| XGBoost | 0.7062 | 0.3763 | 0.2130 | 0.2343 |
| Neural Network (MLP) | 0.6986 | 0.3599 | 0.2168 | 0.2403 |

XGBoost achieves the highest PR-AUC and is selected as the production candidate. Logistic regression, while lower on PR-AUC, is the best calibrated model (ECE = 0.032) and serves as the interpretable baseline.

---

## Business Context

A consumer lender extends personal loans ranging from $1,000 to $40,000. Each approved loan that later defaults generates a net loss equal to the outstanding principal. Each rejected loan from a creditworthy applicant represents forgone interest income. The model's job is to distinguish these two error types and allow the credit risk team to choose an operating threshold that matches their capital allocation strategy.

Two operating points are documented:

| Operating Point | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| F1-optimal | 0.491 | 0.331 | 0.664 | 0.441 |
| Conservative | 0.929 | 1.000 | 0.000 | 0.000 |

The conservative threshold is noted here as a calibration reference point. At the current data distribution, no threshold achieves both precision ≥ 0.80 and meaningful recall — this is a finding in itself and is discussed in the analysis summary.

---

## Data

**Source:** [Lending Club Accepted Loans — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Download `accepted_2007_to_2018Q4.csv.gz` and place it in `data/raw/`. The file is gitignored.

The cleaned dataset contains **1,345,350 loans** with a **20.0% overall default rate**.

| Split | Rows | Default Rate |
|---|---|---|
| Train (pre-2016) | 826,606 | 18.4% |
| Val (2016) | 293,105 | 23.3% |
| Test (2017–2018) | 225,639 | 21.3% |

A temporal split is used throughout. The model is trained on loans originated before 2016 and evaluated on loans from 2017–2018. This is the only split that reflects realistic deployment conditions and is required for regulatory review.

**Key variables used:**

| Variable | Description |
|---|---|
| `loan_status` | Target: Charged Off / Default = 1, Fully Paid = 0 |
| `int_rate` | Loan interest rate |
| `dti` | Debt-to-income ratio |
| `fico` | FICO score midpoint (from fico_range_low + fico_range_high) |
| `annual_inc` | Self-reported annual income (capped at 99th percentile) |
| `loan_amnt` | Requested loan amount |
| `term` | 36 or 60 months |
| `grade` | LC-assigned loan grade (A–G, ordinally encoded) |
| `purpose` | Stated loan purpose |
| `revol_util` | Revolving credit utilization rate |

---

## Feature Selection

Candidate features were screened through four data-driven steps before modeling:

1. **Leakage audit** — all post-origination fields excluded (payment history, recoveries, outstanding principal). `sub_grade` dropped as redundant with `grade`.
2. **Missingness filter** — no columns exceeded the 40% missing threshold after cleaning.
3. **Near-zero variance filter** — no columns flagged.
4. **Redundancy check** — no feature pairs exceeded |r| = 0.90.

Final feature count after encoding: **40 features**.

Top numeric features by absolute correlation with default:

| Feature | |r| with target |
|---|---|
| int_rate | 0.259 |
| term | 0.176 |
| fico | 0.131 |
| dti | 0.085 |
| mort_acc | 0.069 |
| loan_amnt | 0.066 |

---

## Key Findings

**Model selection:** XGBoost is the production candidate. It achieves the highest PR-AUC (0.376) and the best discriminative ability (ROC-AUC 0.706). PR-AUC is the primary metric here because the dataset is imbalanced at 20% — ROC-AUC can be misleading under class imbalance.

**Calibration:** Logistic regression is the best-calibrated model (ECE = 0.032). XGBoost and the neural network both show poor calibration (ECE > 0.23), meaning their raw predicted probabilities are less reliable as risk estimates. Temperature scaling was applied to the neural network; isotonic calibration to logistic regression.

**Feature importance:** The four most important features by SHAP value are interest rate, FICO score, DTI, and revolving utilization — consistent with traditional credit underwriting.

**Threshold:** The F1-optimal threshold of 0.491 achieves precision of 0.331 and recall of 0.664. The absence of a threshold meeting both precision ≥ 0.80 and recall > 0 is a finding: at the current class distribution and feature set, the model cannot simultaneously achieve high precision and meaningful recall. This should inform the credit risk team's expectations before deployment.

**Fairness:** At the F1-optimal threshold, the false positive rate gap across borrower income terciles is 15.8 percentage points (high income: 28.5%, low income: 44.3%). This exceeds the 5-point tolerance defined for this engagement. Low-income borrowers are disproportionately flagged as false positives. This is a material fairness concern and is documented for regulatory review.

| Income Group | Accuracy | FPR | TPR | Selection Rate |
|---|---|---|---|---|
| High income | 0.691 | 0.285 | 0.587 | 0.341 |
| Mid income | 0.640 | 0.368 | 0.667 | 0.430 |
| Low income | 0.597 | 0.443 | 0.720 | 0.511 |

---

## Repository Structure

```
loan-default-risk/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                              # original CSV (gitignored — download from Kaggle)
│   ├── processed/                        # cleaned Parquet and CSV
│   ├── splits/                           # train / val / test Parquet and CSV files
│   └── lending_club_data_dictionary.xlsx # variable definitions from Lending Club
├── notebook/
│   └── loan_default_risk.ipynb
└── reports/
    ├── figures/                          # saved plots
    ├── analysis_summary.md               # analysis report
    └── model_comparison.csv              # model performance table
```

---

## Reproduce the Results

```bash
git clone https://github.com/aanyame22/loan-default-risk.git
cd loan-default-risk
python -m venv venv
source venv/bin/activate        # Mac/Linux
pip install -r requirements.txt
```

Download the Lending Club CSV from Kaggle, place it in `data/raw/`, then run `notebook/loan_default_risk.ipynb` top to bottom. Each section saves its outputs so the notebook can be re-run incrementally.

---

## Limitations

The fairness analysis uses income as a proxy for protected class status. A production system would require explicit demographic data and a formal disparate impact analysis under ECOA and the Fair Housing Act. The FPR gap found here (15.8 points at the F1-optimal threshold) would warrant intervention before deployment.

The temporal test window (2017–2018) coincides with changes in Lending Club's origination standards, which may affect generalizability. Ongoing monitoring of score distributions and default rates is required.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | pandas, numpy, pyarrow |
| ML — baseline | scikit-learn (LogisticRegression, CalibratedClassifierCV) |
| ML — gradient boosting | xgboost |
| ML — neural network | PyTorch |
| Hyperparameter tuning | optuna |
| Explainability | shap |
| Fairness | fairlearn (MetricFrame) |
| Visualization | matplotlib, seaborn |
