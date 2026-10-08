# Datathon Task 1: 80 / 10 / 10 Chronological Retraining Report

## 1. Executive Summary & Split Design
This experiment retrained the complete Task 1 machine learning architecture using a 3-way chronological split to provide **unbiased out-of-sample benchmark metrics** on an untouched holdout test set before generating final competition predictions.

- **80% Training Set:** 73,522 deliveries (2024-01-01 to 2025-09-12)
- **10% Validation Set:** 9,146 deliveries (2025-09-13 to 2025-11-28) — used for early-stopping iteration tuning.
- **10% Holdout Test Set:** 9,226 deliveries (2025-11-29 to 2026-02-14) — untouched benchmark.
- **Competition Test Set:** 5,014 deliveries (predicted using full history refit).

---

## 2. Unbiased Benchmark Performance on 10% Holdout Test

### Target 1: Service Handling Duration (`CatBoostRegressor`)
- **Mean Absolute Error (MAE):** **3.9969 minutes**
- **Median Absolute Error (MedAE):** **2.6229 minutes**
- **Root Mean Squared Error (RMSE):** **7.1369 minutes**
- **Coefficient of Determination ($R^2$):** **79.18%**
- **Optimal Tree Iterations:** 971 rounds

### Target 2: Lateness Probability (`CatBoostClassifier`)
- **ROC-AUC (Discrimination):** **97.35%** (0.9735)
- **Precision-Recall AUC (PR-AUC):** **0.8683**
- **Log Loss:** **0.1416**
- **Brier Score:** **0.0438**
- **Accuracy (0.5 threshold):** **94.02%**
- **Optimal Tree Iterations:** 513 rounds

---

## 3. Probability Calibration Table (10% Holdout Test)

| Probability Bin | Stops | Mean Predicted Probability | Actual Empirical Lateness Rate |
| :---: | :---: | :---: | :---: |
| (-0.001, 0.1] | 6,999 | 0.0097 | 0.94% |
| (0.1, 0.2] | 487 | 0.1465 | 16.43% |
| (0.2, 0.3] | 308 | 0.2464 | 28.25% |
| (0.3, 0.4] | 196 | 0.3477 | 34.69% |
| (0.4, 0.5] | 181 | 0.4519 | 51.38% |
| (0.5, 0.6] | 165 | 0.5478 | 66.06% |
| (0.6, 0.7] | 134 | 0.6468 | 70.15% |
| (0.7, 0.8] | 143 | 0.7526 | 81.82% |
| (0.8, 0.9] | 181 | 0.8504 | 88.95% |
| (0.9, 1.0] | 432 | 0.9657 | 96.30% |

---

## 4. Deliverables in `task1_80_10_10_outputs/`

1. `submission_task1.csv` — Exact competition format submission (5,014 rows).
2. `task1_service_model.cbm` — Final full-data CatBoost service duration regressor.
3. `task1_late_model.cbm` — Final full-data CatBoost lateness probability classifier.
4. `test_holdout_predictions.csv` — Full order-level predictions and actuals on the 10% holdout test set (9,226 rows).
5. `test_holdout_calibration.png` — Calibration curve plot on the holdout set.
6. `test_holdout_service_residuals.png` — Distribution of handling error residuals.
7. `service_feature_importance.png` — Feature importance bar chart for service duration.
8. `late_feature_importance.png` — Feature importance bar chart for lateness probability.
