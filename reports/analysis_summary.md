# Analysis Summary
# Loan Default Risk — Supervised Classification with Fairness Evaluation

**Author:** Adwoa Nyame

---

## Overview

This analysis builds a loan default prediction system for a simulated engagement with a consumer lending institution. Three classifiers are evaluated — logistic regression, XGBoost, and a PyTorch MLP — and compared on predictive accuracy, probability calibration, and fairness across borrower income groups.

The dataset is the Lending Club accepted loans file (2007–2018), containing 1,345,350 resolved loans with a 20% overall default rate. The data is split temporally: training on pre-2016 loans (826,606 rows), validating on 2016 (293,105 rows), and testing on 2017–2018 (225,639 rows). This split prevents data leakage and reflects how the model would perform on loans issued after the training window, which is the only split that matters for regulatory review.

---

## Section 1 — Feature Selection

Candidate features were first screened for origination-time availability. Any field known only after the loan is issued — payment history, collection amounts, outstanding principal — was excluded before modeling began. The remaining columns were then filtered by missingness, near-zero variance, and pairwise redundancy. All four filters passed cleanly: no columns exceeded the 40% missingness threshold, no near-zero variance columns were found, and no feature pairs exceeded |r| = 0.90. The final feature set after one-hot encoding contains 40 features.

The most consequential exclusion was `sub_grade`, which is a finer-grained version of `grade` and was dropped at the leakage audit stage as redundant. Among numeric features, interest rate has the strongest univariate correlation with default (|r| = 0.259), followed by loan term (0.176) and FICO score (0.131). Loan term ranking second — ahead of FICO — is worth noting: 60-month loans default at a meaningfully higher rate than 36-month loans independent of borrower credit quality.

---

## Section 2 — Model Performance

ROC-AUC measures discriminative ability across all thresholds. PR-AUC is the primary metric here because the dataset is imbalanced at 20% — ROC-AUC can look flattering even when a model struggles to distinguish defaults from fully-paid loans at practical thresholds. Brier score measures the accuracy of the probability estimates themselves. ECE measures calibration quality: a perfectly calibrated model has ECE = 0.

| Model | ROC-AUC | PR-AUC | Brier | ECE | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.6958 | 0.3607 | 0.1557 | 0.0320 | 0.0959 | 0.4978 | 0.0531 |
| XGBoost | 0.7062 | 0.3763 | 0.2130 | 0.2343 | 0.4407 | 0.3346 | 0.6453 |
| Neural Network | 0.6986 | 0.3599 | 0.2168 | 0.2403 | 0.4362 | 0.3282 | 0.6499 |

XGBoost achieves the highest PR-AUC (0.376) and the best discriminative performance (ROC-AUC 0.706), confirming it as the production candidate. Logistic regression has the lowest Brier score (0.156) and by far the best calibration (ECE = 0.032), meaning its predicted probabilities are the most accurate as actual risk estimates. XGBoost and the neural network both show poor calibration (ECE > 0.23), which means their raw probabilities should not be used as standalone risk scores without post-hoc calibration.

The neural network stopped early at epoch 6 and performs similarly to XGBoost on recall but with marginally lower precision. It offers no meaningful advantage over XGBoost on this tabular dataset and introduces substantially more complexity.

The choice between XGBoost and logistic regression involves a tradeoff: XGBoost is the better discriminator; logistic regression is the better probability estimator. For a deployment where the model output is used as a risk score rather than a binary flag, logistic regression's calibration advantage is significant.

---

## Section 3 — Feature Importance

SHAP values from the XGBoost model identify which features drive predictions and in which direction. Based on both the univariate correlations from feature selection and the SHAP output, the most important features are interest rate, loan term, FICO score, and DTI. This is broadly consistent with traditional credit underwriting, which is reassuring from a face validity standpoint.

The direction of effects is intuitive: higher interest rates and longer terms are associated with higher default risk; higher FICO scores reduce it. Interest rate is both the strongest univariate predictor and the top SHAP feature, which reflects the fact that Lending Club's own grade-based pricing already encodes much of the credit risk signal — borrowers assigned higher rates are riskier by design.

---

## Section 4 — Threshold Selection

The default 0.5 threshold is rarely the right operating point for imbalanced credit risk problems. The threshold sweep results are shown below.

| Operating Point | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| F1-optimal | 0.491 | 0.331 | 0.664 | 0.441 |
| Conservative | 0.929 | 1.000 | 0.000 | 0.000 |

The F1-optimal threshold of 0.491 achieves precision of 0.331 and recall of 0.664. This is the recommended operating point: at this threshold, the model flags approximately 66% of actual defaults at a cost of roughly two false alarms for every true positive.

The conservative threshold result — precision of 1.000 and recall of 0.000 at threshold 0.929 — is a degenerate outcome. It means no threshold in the sweep simultaneously achieves precision above 80% with any meaningful recall. This is an honest finding about the limits of the current model, not a usable operating point. The practical implication is that the model is not well-suited for a deployment requiring both high precision and substantial coverage. The credit risk team should treat the F1-optimal threshold as the sole actionable operating point until the model is retrained on more recent data or with additional features.

The model produces a probability; the threshold converts it into an action. That conversion is a business decision, but the options here are constrained.

---

## Section 5 — Fairness Evaluation

Lending Club does not include race or gender in its public dataset, so borrower annual income terciles serve as the primary demographic grouping. This mirrors regulatory practice under ECOA, where income can serve as a proxy variable for protected class membership in disparate impact analysis.

At the F1-optimal threshold, the false positive rate gap across income groups is 15.8 percentage points: high-income borrowers have an FPR of 28.5%, while low-income borrowers have an FPR of 44.3%. This means a creditworthy low-income borrower is flagged as a default risk at a rate 58% higher than a creditworthy high-income borrower with the same actual outcome. This gap substantially exceeds the 5-point tolerance defined for this engagement and is a material fairness concern.

| Income Group | Accuracy | FPR | TPR | Selection Rate |
|---|---|---|---|---|
| High income | 0.6910 | 0.2854 | 0.5865 | 0.3409 |
| Mid income | 0.6395 | 0.3676 | 0.6666 | 0.4301 |
| Low income | 0.5966 | 0.4434 | 0.7199 | 0.5112 |

This finding is not unusual in credit risk models. Low-income borrowers tend to have higher DTI ratios and lower FICO scores on average, and the model amplifies these differences into disparate error rates. The FPR gap chart in section 13 of the notebook shows that the gap exceeds 5 points across nearly the entire threshold range, meaning threshold adjustment alone cannot resolve the disparity. Reweighting the training data or applying a fairness constraint during training would be required before this model could be deployed in a regulated context.

If a regulator asks whether the model treats borrowers equally, the answer here is no. That answer requires specifying a threshold, and at the only practical operating point the disparity is large.

---

## Section 6 — Limitations and Extensions

The fairness analysis uses income as a proxy for protected class status because direct demographic variables are not available in the public Lending Club dataset. A production system would require explicit demographic data and a formal disparate impact analysis under ECOA and the Fair Housing Act. The 15.8-point FPR gap found here would constitute a prima facie disparate impact finding under the standard four-fifths rule and would require remediation.

The temporal test window (2017–2018) coincides with changes in Lending Club's origination standards. The default rate rises from 18.4% in the training window to 23.3% in validation, which suggests the model is tested on a harder distribution than it was trained on. This may be partially responsible for the limited precision achieved at any threshold.

The neural network early stopping at epoch 6 indicates the model did not fully converge, likely due to the large dataset size relative to the number of training epochs. Increasing patience or reducing the learning rate could improve performance.

Extensions that would meaningfully strengthen this system include: post-hoc calibration of the XGBoost model before using its probabilities as risk scores; applying a fairness constraint during XGBoost training to reduce FPR disparity; adding macroeconomic features to improve temporal generalizability; and building a model card documenting intended use, performance bounds, and failure modes for the review team.

---

## Section 7 — Recommendations

**Model selection:** Use XGBoost for ranking and flagging. Apply post-hoc isotonic calibration before using its outputs as probability estimates. If calibrated probabilities are the primary output — for example, for pricing or reserves — logistic regression is the more defensible choice given its ECE of 0.032.

**Operating threshold:** Use the F1-optimal threshold of 0.491. No threshold achieves high precision with meaningful recall on this dataset; this is the only operating point that captures a useful fraction of defaults.

**Fairness:** The 15.8-point FPR gap across income groups at the recommended threshold is too large for deployment in a regulated context. Before moving to production, retrain the model with a fairness constraint or apply post-processing threshold adjustments per income group. Report FPR by income group at every model review cycle.

**Monitoring:** Track precision and default rate among flagged loans monthly. Given the shift in default rates between the training (18.4%) and validation (23.3%) windows, the model should be retrained on data from 2016 onward before deployment and refreshed annually.
