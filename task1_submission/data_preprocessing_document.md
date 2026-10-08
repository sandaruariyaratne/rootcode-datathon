# Datathon Task 1: Data Preprocessing & Methodology Document

## 1. Executive Summary & Objective
Waypoint Logistics operates an urban multi-temperature retail delivery network across Western and Central Sri Lanka (Peliyagoda and Kandy depots). For every planned delivery in the test dataset (`task1_test_inputs.csv`, 5,014 deliveries), the objective is to predict:
1. **`pred_service_min`:** Handling and unloading duration at the destination outlet in minutes.
2. **`pred_late_prob`:** Calibrated probability that the delivery vehicle arrives after the outlet's delivery window closes.

This document details the data preparation, label construction, leakage prevention firewall, cleaning procedures, and feature engineering rationale.

---

## 2. Dataset Merging & Entity Alignment
The raw datasets represent multi-table relational operational logs. They were joined with strict 1:1 integrity assertions:

1. **Order Manifest to Route Leg Matching:**
   Each row in `deliveries_train.csv` denotes an order stop with a unique `delivery_id`. Each dispatched delivery was mapped to its exact route segment in `route_legs_train.csv` via composite key `(route_id, seq_in_route) == (route_id, seq)`.
2. **Reference Table Merging:**
   - **Outlets (`outlets.csv`):** Matched on `outlet_id` to attach physical infrastructure constraints (`dock_type`, `parking_constraint`, `mall_window`).
   - **Vehicles (`vehicles.csv`):** Matched on `vehicle_id` to attach physical carrying bounds (`weight_cap_kg`, `volume_cap_m3`, `fuel_type`).
   - **District Travel (`district_travel.csv`):** Matched on `(district, depot)` to attach baseline travel times and inter-stop free-flow speeds.
   - **Service Allowance (`service_allowance.csv`):** Matched on `(brand, dock_type)` to attach dispatcher baseline unloading allowances.
   - **Calendar (`calendar.csv`):** Matched on `date` to extract day-of-week, festival ramp demand surges, paydays, and operating status.
   - **Road Conditions (`road_conditions.csv`):** Matched on `(district, date)` to attach date-specific disruptions (monsoon floods, roadworks).
   - **Traffic Congestion (`traffic_speed.csv`):** Matched on `(district, planned_depart_hour, monsoon)` to account for time-of-day traffic speed degradation.

---

## 3. Ground Truth Label Construction (Strict Zero-Leakage)

Historical actual execution times are available exclusively in the training route logs. Targets were constructed following competition guidelines:

### A. Handling Service Time (`target_service_min`)
If a delivery vehicle arrives before the customer outlet opens, the driver waits outside the gate. Under the competition rules, **early waiting time must not be counted as handling time**:
$$\text{service\_start\_min} = \max(\text{actual\_arrival\_min}, \text{window\_open\_min})$$
$$\text{target\_service\_min} = \text{leave\_outlet\_min} - \text{service\_start\_min}$$

*Verification:*
- All resulting values are non-negative ($\ge 0$).
- Orders that arrived prior to store opening correctly have their dwell time truncated to begin when the store window opens.

### B. Arrival Lateness Flag (`target_late`)
A delivery is defined as late if and only if the vehicle arrives strictly after the window closure:
$$\text{target\_late} = \begin{cases} 1 & \text{if } \text{actual\_arrival\_min} > \text{window\_close\_min} \\ 0 & \text{otherwise} \end{cases}$$
An arrival exactly at the closing minute is classified as on-time.

---

## 4. Anti-Leakage Firewall
In a production deployment, predictions must be generated before vehicles leave the depot. Therefore, all actual execution telemetry columns were firewalled out of model features:
* **Forbidden Columns:** `actual_depart_time`, `actual_travel_duration_min`, `arrival_time`, `leave_outlet_time`, `actual_arrival_min`, `leave_outlet_min`, `service_start_min`.
* **Guaranteed Inputs:** Only pre-dispatch planned variables, schedule constraints, reference dimensions, and environmental signals are utilized.

---

## 5. Feature Engineering Rationale

We developed 60 structured features across 5 domain categories:

### 1. Order Physics & Handling Workload
- `order_weight_kg`, `order_volume_m3`, `order_units`: Direct physical cargo dimensions.
- `weight_per_unit`, `volume_per_unit`: Distinguishes bulk heavy pallets from lightweight bulky parcels.
- `density_kg_m3`: Order density ($\text{kg} / \text{m}^3$) is a key driver of unloading equipment requirements.
- `weight_utilisation`, `volume_utilisation`: Capacity fill ratios relative to vehicle limits.

### 2. Schedule Buffer & Window Geometry (Primary Lateness Drivers)
- `planned_slack_min` $= \text{window\_close\_min} - \text{planned\_arrival\_min}$: The critical buffer. Small or negative planned slack indicates high risk of lateness.
- `planned_early_gap_min` $= \text{window\_open\_min} - \text{planned\_arrival\_min}$: Captures early arrival waiting potential.
- `window_width_min` $= \text{window\_close\_min} - \text{window\_open\_min}$: Tight windows provide less margin for delivery delays.

### 3. Route Structure & Accumulated Journey Delay
- `seq_in_route`, `stops_before`: Sequence position in the multi-stop manifest.
- `route_progress` $= (\text{seq\_in\_route} + 1) / \text{route\_n\_stops}$: Relative position along the route.
- `cum_planned_travel_to_stop_min`: Accumulated driving time from depot.
- `planned_service_before_min`: Cumulative planned handling time at all preceding stops. Delays early in the route cascade forward to downstream deliveries.
- `planned_route_work_to_arrival_min`: Total accumulated workload prior to reaching this delivery.

### 4. Congestion & Environmental Friction
- `traffic_factor` $= 100 / \text{speed\_index}$: Congestion factor by district and hour.
- `disruption_factor` $= 100 / \text{disruption\_index}$: Local roadwork/flooding friction.
- `traffic_adjusted_leg_min`: Planned driving time adjusted by dynamic traffic speeds.
- `traffic_road_adjusted_leg_min`: Combined traffic and disruption travel duration.
- `monsoon`: Extreme weather indicator known to disrupt logistics in Sri Lanka.

### 5. Facility Access & Calendar Constraints
- `dock_type` (`rear_dock`, `street`, `forklift`): Dictates unloading speed and physical bottleneck.
- `parking_constraint` (`van_only`, `standard`): Identifies narrow streets with restricted maneuvering space.
- `mall_window`: Stricter compliance rules and basement dock queuing times.
- `festival_ramp`: Pre-festival delivery surges in Sri Lanka create long dock queues.

---

## 6. Categorical Handling & Missing Value Treatment
- **Categorical Variables:** Handled directly via **CatBoost** native categorical encoding (ordered target statistics and feature combinations), preserving natural grouping without high-dimensional one-hot sparsity. Missing categorical entries were imputed to `"Unknown"`.
- **Numerical Missingness:** Handled natively by gradient boosted trees without artificial zero-imputation that could distort physical ratios.
