# Waypoint Logistics — Datathon Task 1 Submission Package

**Rootcode Tech-Triathlon 2026**  
**Task 1: Predict Handling Service Time and Arrival Lateness Probability**

This folder contains the complete, self-contained submission deliverables for **Datathon Task 1**, structured in full compliance with the requirements set forth in the *Challenge Booklet* (Pages 16, 22, and 23).

---

## 1. Submission Deliverables Summary

| File | Challenge Booklet Requirement | Description |
| :--- | :--- | :--- |
| **`submission_task1.csv`** | **Primary Prediction File (Page 16, 22)** | Exactly 5,014 test deliveries. Preserves original row sequence and identifiers. Includes `pred_service_min` and `pred_late_prob`. |
| **`task1_service_model.cbm`** | **Model File (Page 22)** | Serialized CatBoost handling time regressor. |
| **`task1_late_model.cbm`** | **Model File (Page 22)** | Serialized CatBoost arrival lateness probability classifier. |
| **`Task1_FinalNotebook.ipynb`**| **Final Notebook (Page 22)** | End-to-end reproducible notebook with label construction, preprocessing, training, evaluation, and model loading demonstration. |
| **`data_preprocessing_document.md`** | **Preprocessing Document (Page 22)** | Detailed write-up covering data wrangling, anti-leakage firewall, label mathematics, and feature engineering rationale. |
| **`architecture_and_pipeline.md`** | **Architecture Diagrams (Page 22)** | High-level system architecture, loss formulations, and TMS deployment strategy. |
| **`pipeline_architecture_diagram.png`** | **Architecture Diagram Image (Page 22)** | High-resolution infographic of the end-to-end system architecture. |
| **`ai_tool_disclosure.md`** | **AI Tool Disclosure (Page 22)** | Explicit declaration of human-led vs. AI-assisted work in compliance with competition integrity rules. |
| **`unbiased_metrics_report.md`** | **Evaluation Report (Page 22)** | Benchmark performance on the 10% untouched holdout test set (9,226 orders) with decile calibration table. |
| **`test_holdout_calibration.png`** | **Diagnostic Visual** | Probability calibration curve on the holdout test set. |
| **`test_holdout_service_residuals.png`** | **Diagnostic Visual** | Service handling time error residual distribution. |
| **`service_feature_importance.png`** | **Diagnostic Visual** | Top 15 predictive drivers for handling duration. |
| **`late_feature_importance.png`** | **Diagnostic Visual** | Top 15 predictive drivers for arrival lateness. |

---

## 2. Benchmark Performance Summary (10% Untouched Holdout Test Set)

Evaluated on 9,226 out-of-sample deliveries (`2025-11-29` to `2026-02-14`):

* **Service Handling Time Model:**
  - **MAE:** **3.9969 minutes**
  - **MedAE:** **2.6229 minutes**
  - **RMSE:** **7.1369 minutes**
  - **$R^2$:** **79.18%**
* **Arrival Lateness Probability Model:**
  - **ROC-AUC:** **97.35%** (0.9735)
  - **PR-AUC:** **0.8683**
  - **Log Loss:** **0.1416**
  - **Brier Score:** **0.0438**
  - **Accuracy (0.5 threshold):** **94.02%**

---

## 3. Strict Verification & Format Checks

The submission file `submission_task1.csv` has been programmatically validated against all official constraints:
- [x] Exactly **5,014 rows** (matches `task1_test_inputs.csv` 1:1).
- [x] Exact column headers: `delivery_id`, `pred_service_min`, `pred_late_prob`.
- [x] Preserves original `delivery_id` ordering (`ORD0092308` to `ORD0097345`).
- [x] Zero null or NaN values.
- [x] `pred_service_min` values are non-negative ($\ge 0.0$).
- [x] `pred_late_prob` values are strictly bounded in $[0.0, 1.0]$.
- [x] Models loaded from disk reproduce predictions seamlessly.
