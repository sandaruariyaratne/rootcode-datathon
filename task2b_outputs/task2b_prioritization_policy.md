# Task 2B Prioritization Policy — Scenario S1

## Allocation approach
We treated the peak-day plan as a constrained fleet-allocation problem. Hard feasibility rules were enforced before prioritization: each trip contains one brand and one district; chilled orders use reefer vehicles; van-only outlets use vans; vehicles serve only their home depot; whole orders are not split; every trip stays within both weight and volume capacity; each vehicle runs no more than two trips; and the Fresh and Style/Tech daily time budgets are respected using the published district travel and service-allowance tables.

## Priority policy
Our primary objective was to maximize the number of orders served. Among allocations with the same order throughput, we preferred orders that had already been deferred, outlets with more days since last service, chilled orders, van-only orders, Fresh orders in the festival-ramp scenario, and orders with fewer compatible available vehicles. The throughput objective was made lexicographically dominant, so priority bonuses could not reduce the maximum number of served orders merely to favor a higher-priority order.

## Result
The generated plan serves **79 of 85 orders** and defers **6**. The deferred demand represents approximately **9,770.9 kg** and **81.566 m³**. **1** deferred orders had also been deferred yesterday.

The scenario had **28 available vehicles**, including **4 reefers** and **3 vans**, against **26 chilled orders** and **6 van-only orders**. These constrained vehicle classes were therefore protected from being used unnecessarily on orders that could be handled by less specialized vehicles.

## Feasibility and deferral interpretation
All generated trips passed an independent validation of the published Task 2B hard rules. The solver terminated successfully within the configured tolerance; the reported MIP gap was 1.99%. The allocation is fully validated for feasibility, but we do not describe it as an exact zero-gap proof.

The deferred-order table should be reviewed together with the trip summary when explaining which deferrals were driven by scarce vehicle compatibility, capacity, trip-time budgets, or the choice to preserve higher-priority service.