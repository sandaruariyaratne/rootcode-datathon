# Rootcode Tech-Triathlon 2026 — Datathon Submissions

**Waypoint Logistics Optimization Platform**

This repository contains the verified, reproducible submission deliverables for **Datathon Task 1** and **Datathon Task 2B**.

---

## Repository Structure

```text
rootcode-datathon/
├── task1_submission/                      # Task 1 Deliverables (Predictive ML Pipeline)
│   ├── submission_task1.csv              # Official predictions (5,014 test deliveries)
│   ├── task1_service_model.cbm           # Handling duration CatBoost regressor
│   ├── task1_late_model.cbm              # Arrival lateness CatBoost classifier
│   ├── Task1_FinalNotebook.ipynb         # End-to-end reproducible training & inference notebook
│   ├── pipeline_architecture_diagram.png # 4K UHD system & deployment architecture diagram
│   ├── architecture_and_pipeline.md      # Architecture documentation & loss formulations
│   ├── data_preprocessing_document.md    # Preprocessing, label logic & anti-leakage firewall
│   ├── unbiased_metrics_report.md        # Benchmark report on 10% holdout test set (3.99 min MAE, 97.35% AUC)
│   ├── ai_tool_disclosure.md             # AI tool usage disclosure statement
│   ├── test_holdout_calibration.png      # 10-decile probability calibration curve
│   ├── test_holdout_service_residuals.png# Handling duration error residual distribution
│   ├── service_feature_importance.png    # Top predictive features for service duration
│   ├── late_feature_importance.png       # Top predictive features for arrival lateness
│   └── README.md                         # Detailed Task 1 reviewer guide
│
└── task2b_outputs/                        # Task 2B Deliverables (Peak-Day Fleet Allocation)
    ├── submission_task2b.csv             # Official allocation decisions (86 orders across S1 & S2)
    ├── task2b_prioritization_policy.md   # Mathematical prioritization framework & dual-trip policy
    ├── task2b_allocation_overview.png    # Visual dashboard of vehicle assignments & capacity
    ├── task2b_trip_summary.csv           # Trip-level utilization & feasibility metrics
    └── task2b_deferred_orders.csv        # Detailed audit of deferred low-margin non-perishable orders
```

---

## 1. Task 1: Handling Duration & Lateness Probability

* **Objective:** Predict service handling minutes ($\hat{y} \ge 0$) and arrival lateness probability ($\hat{p} \in [0, 1]$) for 5,014 test orders without data leakage.
* **Evaluation Metrics (10% Untouched Chronological Holdout Set):**
  * **Handling Service Duration:** **3.99 min MAE**, 7.12 min RMSE ($R^2 = 0.810$)
  * **Arrival Lateness Probability:** **97.35% ROC-AUC**, 93.9% Accuracy, 0.169 LogLoss
  * **Probability Calibration:** 10-Decile Reliability $R^2 = 0.992$, Brier Score = $0.046$
* **Runtime Efficiency:** $<0.05$ ms / delivery CPU inference latency ($<250$ ms for 5,000 stops, zero GPU dependency).
* **Navigation:** See [`task1_submission/README.md`](task1_submission/README.md) for complete details.

---

## 2. Task 2B: Peak-Day Fleet Allocation & Prioritization

* **Objective:** Maximize high-margin, perishable, and festival order fulfillment during severe fleet capacity bottlenecks under strict 2-trip limits and vehicle volume constraints.
* **Allocation Performance:**
  * **Scenario 1 (Peliyagoda Surge):** 93.8% fulfillment (30/32 orders served), zero SLA or capacity breaches.
  * **Scenario 2 (Extreme Network Stress):** 83.3% fulfillment (45/54 orders served), 100% chilled/perishable protection.
* **Navigation:** See [`task2b_outputs/task2b_prioritization_policy.md`](task2b_outputs/task2b_prioritization_policy.md) for full policy logic.
