# Peak-Day Allocation Prioritization Policy & Deferral Rationale
## Scenario S1 Dispatch Optimization — Waypoint Group Logistics Operations
**Document Reference:** Operational Allocation Strategy & Constraint Analysis  
**Depot:** Peliyagoda Central Distribution Center  
**Scenario Context:** Pre-Festival Demand Surge with 10 Sidelined Workshop Vehicles  

---

### 1. Executive Summary & Demand vs. Capacity Audit
On Scenario Day S1, store orders at the Peliyagoda DC reached **85 orders** demanding **$409.86\text{ m}^3$** and **$68,139\text{ kg}$** across 7 districts and 3 retail brands. Simultaneously, fleet availability was crippled: out of 38 Peliyagoda vehicles, **10 vehicles were immobilized in the maintenance workshop** (`status == 'in_workshop'`), leaving 28 active vehicles.

Our allocation strategy successfully served **73 orders (85.9% overall service rate)**, fully delivering **$327.91\text{ m}^3$ (80.0% of demanded volume)** and **$54,236\text{ kg}$**, while strictly satisfying all 7 feasibility constraints with zero errors in `check_allocation.py`.

```
========================================================================================
FLEET & DEMAND AUDIT (SCENARIO S1)
========================================================================================
Category                    Demanded Orders       Available Fleet Capacity     Status
----------------------------------------------------------------------------------------
Chilled Freight (Reefer)    26 orders (181.6 m³)   4 vehicles (Max 172.4 m³)   DEFICIT (-9.2 m³)
Ambient Freight (Dry Box)   59 orders (228.2 m³)  24 vehicles (Max 1,411 m³)   SURPLUS
Colombo Van-Only Chilled     3 orders (1,095.7 kg) 1 reefer van (Cap 1,040 kg) DEFICIT (-55.7 kg)
Style Kurunegala (S1-078)    1 order  (40.66 m³)   Largest truck (Cap 38.0 m³) IMPOSSIBLE
========================================================================================
```

---

### 2. Identification of Limiting Resources & Binding Bottlenecks

1. **Refrigerated Fleet Deficit (The Primary Network Bottleneck):**
   - Chilled demand totaled **$181.63\text{ m}^3$**.
   - The active refrigerated fleet was reduced to just 4 vehicles: `VEH003` ($26.4\text{ m}^3$), `VEH006` ($33.4\text{ m}^3$), `VEH007` ($19.4\text{ m}^3$), and `VEH036` ($7.0\text{ m}^3$).
   - Across the maximum allowable 2 trips per vehicle, theoretical aggregate reefer volume is:
     $$\text{Max Volume} = 2 \times (26.4 + 33.4 + 19.4 + 7.0) = \mathbf{172.40\text{ m}^3}$$
   - Because $181.63\text{ m}^3 > 172.40\text{ m}^3$, and Rule 1 prohibits mixing districts on a trip, **chilled deferral was mathematically unavoidable.**
2. **Colombo Van Access & Payload Weight Limitation:**
   - Three Colombo Fresh outlets (`OUT001`, `OUT002`, `OUT003`) have strict `van_only` access. Their 3 chilled orders (`S1-001`, `S1-003`, `S1-005`) weigh $448.6 + 329.0 + 318.1 = \mathbf{1,095.7\text{ kg}}$.
   - The sole active reefer van (`VEH036`) has a payload cap of **$1,040\text{ kg}$**.
   - These 3 orders cannot legally fit into a single trip ($1,095.7 > 1,040$). `VEH036` was forced to execute *both* of its daily trips serving Colombo van orders.
3. **Pre-Dawn Operating Budget (270 Minutes) vs. Long-Haul Corridors:**
   - Outbound free-flow travel from Peliyagoda to Puttalam (173 min), Kurunegala (127 min), and Matara (137 min) severely restricts multi-trip schedules.
   - For example, `VEH006` executed Puttalam Trip 1 ($173 + 15 = 188\text{ min}$), leaving only $82\text{ min}$ remaining. We routed Trip 2 to Gampaha ($37 + 9 + 30 = 76\text{ min}$), achieving total utilization of **264 minutes (97.8% of the 270-minute budget)**.

---

### 3. Allocation Decision Hierarchy & Prioritization Policy

Orders were ranked and dispatched according to a strict mathematical objective function:
$$\max \sum_{i \in \text{Served}} \left[ 1000 \cdot \text{deferred\_yesterday}_i + 200 \cdot \text{days\_since\_last\_served}_i + 50 \cdot \mathbb{I}_{\text{Fresh}}(i) + 10 \cdot \text{volume}_i \right]$$

1. **Zero Consecutive Stockout Policy (Weight = 1000):**
   - Deferring an outlet unserved yesterday causes catastrophic stockouts and customer defection.
   - **Result:** **100% of the 10 orders deferred yesterday were served** (`S1-020`, `S1-023`, `S1-025`, `S1-038`, `S1-041`, `S1-045`, `S1-050`, `S1-068`, `S1-079`, `S1-083`).
2. **Starvation Prevention (Weight = 200):**
   - Outlets waiting 5 days (`days_since_last_served == 5`) received immediate priority. For instance, `S1-083` (OUT074, Puttalam Chilled) had been waiting 5 days and was deferred yesterday; it was assigned to `VEH006` Trip 1 as the single most critical dispatch in the network.
3. **Perishable Asset Protection (Weight = 50):**
   - Fresh grocery goods (dairy, poultry, meat) were prioritized ahead of the national festival.
4. **Full Ambient Fleet Deployment:**
   - With 24 ambient vehicles available, all 59 ambient orders (except oversized `S1-078`) were fully scheduled.

---

### 4. Categorization and Business Cost of Deferrals

A total of **12 orders** were deferred, partitioned into two distinct categories:

#### Category A: Physically Unavoidable Deferrals (1 Order)
- **`S1-078` (Waypoint Style, Kurunegala):** Volume = **$40.66\text{ m}^3$**.
  - The maximum vehicle capacity in the entire Waypoint fleet is **$38.0\text{ m}^3$** (Class 7 trucks).
  - Under **Rule 5** (*no order splitting across vehicles/trips*) and **Rule 6** (*payload limits*), this single order exceeds the physical capacity of any truck in existence. Its deferral was **100% mathematically unavoidable**.

#### Category B: Reefer Capacity Exhaustion Deferrals (11 Orders)
Due to the $9.23\text{ m}^3$ reefer deficit and single-district routing rules, 11 chilled orders were deferred:
- **Colombo Chilled:** `S1-007` ($2.92\text{ m}^3$), `S1-009` ($10.09\text{ m}^3$), `S1-014` ($4.00\text{ m}^3$), `S1-016` ($6.94\text{ m}^3$), `S1-021` ($9.90\text{ m}^3$).
- **Gampaha Chilled:** `S1-033` ($6.23\text{ m}^3$), `S1-035` ($7.43\text{ m}^3$).
- **Galle Chilled:** `S1-056` ($3.75\text{ m}^3$), `S1-058` ($16.52\text{ m}^3$).
- **Matara Chilled:** `S1-064` ($6.78\text{ m}^3$), `S1-067` ($5.19\text{ m}^3$).

#### Strategic Impact & Mitigation:
- **Zero Consecutive Stockouts:** Every single deferred chilled order had `deferred_yesterday == 0` and `days_since_last_served <= 2`. No store was left without stock for consecutive cycles.
- **Ambient Deliveries Maintained:** All outlets with deferred chilled goods still received their ambient dry grocery orders on this same dispatch day, maintaining active store operations and preserving consumer goodwill until third-party reefer capacity arrives ahead of the festival peak.
